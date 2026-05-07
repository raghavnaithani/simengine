"""Telemetry and metrics collection for quality monitoring.

Records per-job metrics to MongoDB for tracking quality trends and alerting on regressions.
"""

import asyncio
from collections import Counter
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from backend.app.database.connection import get_database
from backend.app.utils.logger import append_log, record_event
import statistics


class TelemetryCollector:
    """Collects and stores quality/performance metrics per job."""

    @staticmethod
    def _mean(values: List[float]) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _sanitize_average(metrics_list: List[Dict[str, Any]], key: str) -> float:
        values = [float(metric.get(key, 0)) for metric in metrics_list if isinstance(metric.get(key, 0), (int, float))]
        return TelemetryCollector._mean(values)

    @staticmethod
    def _classify_failure_reason(job: Dict[str, Any]) -> str:
        error_text = str(job.get("error") or job.get("status") or "").lower()
        if "timeout" in error_text:
            return "timeout"
        if "fallback" in error_text:
            return "fallback"
        if "citation" in error_text or "provenance" in error_text:
            return "citation"
        if "llm" in error_text or "ollama" in error_text or "model" in error_text:
            return "model"
        return "other"

    @staticmethod
    def evaluate_alerts(
        metrics_list: List[Dict[str, Any]],
        jobs_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return threshold-based alert payloads for the dashboard."""
        alerts: List[Dict[str, Any]] = []
        if len(metrics_list) < 2:
            return alerts

        latest = metrics_list[0]
        history = metrics_list[1:]

        alert_rules = [
            ("citation_rate", 0.2, 0.5, "citation_rate_drop", "drop"),
            ("diversity_score", 0.15, 0.45, "diversity_drop", "drop"),
            ("error_rate", 0.15, 0.2, "fallback_spike", "rise"),
        ]

        for key, delta_threshold, floor, alert_name, direction in alert_rules:
            latest_value = float(latest.get(key, 0) or 0)
            history_values = [float(metric.get(key, 0) or 0) for metric in history]
            history_mean = TelemetryCollector._mean(history_values)
            if history_mean <= 0:
                continue
            if direction == "drop" and history_mean > floor and (history_mean - latest_value) > delta_threshold:
                alerts.append(
                    {
                        "type": alert_name,
                        "metric": key,
                        "latest": round(latest_value, 3),
                        "baseline": round(history_mean, 3),
                    }
                )
            if direction == "rise" and latest_value >= floor and (latest_value - history_mean) > delta_threshold:
                alerts.append(
                    {
                        "type": alert_name,
                        "metric": key,
                        "latest": round(latest_value, 3),
                        "baseline": round(history_mean, 3),
                    }
                )

        latest_latency = float(latest.get("latency_ms", 0) or 0)
        latency_history = [float(metric.get("latency_ms", 0) or 0) for metric in history]
        latency_baseline = TelemetryCollector._mean(latency_history)
        if latency_baseline > 0 and latest_latency > max(latency_baseline * 1.5, latency_baseline + 1000.0):
            alerts.append(
                {
                    "type": "latency_spike",
                    "metric": "latency_ms",
                    "latest": round(latest_latency, 3),
                    "baseline": round(latency_baseline, 3),
                }
            )

        if jobs_list:
            timeout_jobs = [job for job in jobs_list if "timeout" in str(job.get("error") or job.get("status") or "").lower()]
            timeout_rate = len(timeout_jobs) / len(jobs_list)
            if timeout_rate >= 0.2:
                alerts.append(
                    {
                        "type": "timeout_spike",
                        "metric": "job_timeout_rate",
                        "latest": round(timeout_rate, 3),
                        "baseline": 0.0,
                    }
                )

        return alerts

    @staticmethod
    async def record_job_metrics(
        job_id: str,
        citation_rate: float,
        diversity_score: float,
        quality_score: float,
        latency_ms: float,
        error_rate: float,
        alternatives_count: int,
        title_novelty: float,
        risk_specificity: float,
    ) -> None:
        """Record comprehensive metrics for a job.

        Args:
            job_id: Job identifier
            citation_rate: Fraction of nodes with citations (0-1)
            diversity_score: Avg novelty across nodes (0-1)
            quality_score: Avg quality across nodes (0-1)
            latency_ms: Total job duration in ms
            error_rate: Fraction of nodes created via fallback (0-1)
            alternatives_count: Avg alternatives per node
            title_novelty: Avg title novelty score (0-1)
            risk_specificity: Avg risk specificity score (0-1)
        """
        db = await get_database()
        telemetry = db["telemetry_metrics"]

        metric_doc = {
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc),
            "citation_rate": float(citation_rate),
            "diversity_score": float(diversity_score),
            "quality_score": float(quality_score),
            "latency_ms": float(latency_ms),
            "error_rate": float(error_rate),
            "alternatives_count": float(alternatives_count),
            "title_novelty_score": float(title_novelty),
            "risk_specificity_score": float(risk_specificity),
        }

        await telemetry.insert_one(metric_doc)
        append_log(f"Telemetry: recorded metrics for job {job_id}")

    @staticmethod
    async def get_metrics_summary(limit: int = 20) -> Dict[str, Any]:
        """Get summary of last N jobs' metrics for dashboard.

        Args:
            limit: Number of recent jobs to include

        Returns:
            Summary dict with averages and trends
        """
        db = await get_database()
        telemetry = db["telemetry_metrics"]

        metrics_list = (
            await telemetry.find().sort("timestamp", -1).limit(limit).to_list(None)
        )

        if not metrics_list:
            return {
                "total_jobs_sampled": 0,
                "averages": {},
                "status": "No metrics recorded yet",
            }

        averages = {
            "citation_rate": round(TelemetryCollector._sanitize_average(metrics_list, "citation_rate"), 3),
            "diversity_score": round(TelemetryCollector._sanitize_average(metrics_list, "diversity_score"), 3),
            "quality_score": round(TelemetryCollector._sanitize_average(metrics_list, "quality_score"), 3),
            "latency_ms": round(TelemetryCollector._sanitize_average(metrics_list, "latency_ms"), 3),
            "error_rate": round(TelemetryCollector._sanitize_average(metrics_list, "error_rate"), 3),
            "alternatives_count": round(TelemetryCollector._sanitize_average(metrics_list, "alternatives_count"), 3),
            "title_novelty_score": round(TelemetryCollector._sanitize_average(metrics_list, "title_novelty_score"), 3),
            "risk_specificity_score": round(TelemetryCollector._sanitize_average(metrics_list, "risk_specificity_score"), 3),
        }

        alerts = TelemetryCollector.evaluate_alerts(metrics_list)
        alert = ""
        if alerts:
            alert = "; ".join(
                f"{entry['type']} ({entry['metric']} {entry['latest']:.3f} vs {entry['baseline']:.3f})"
                for entry in alerts[:3]
            )

        return {
            "total_jobs_sampled": len(metrics_list),
            "averages": averages,
            "alert": alert,
            "alerts": alerts,
            "status": "ok",
        }

    @staticmethod
    async def get_dashboard_summary(limit: int = 20) -> Dict[str, Any]:
        """Return a combined quality, grounding, latency, and alert dashboard."""
        db = await get_database()
        telemetry = db["telemetry_metrics"]
        jobs = db["jobs"]

        metrics_list = (
            await telemetry.find().sort("timestamp", -1).limit(limit).to_list(None)
        )

        if not metrics_list:
            return {
                "total_jobs_sampled": 0,
                "status": "No metrics recorded yet",
                "quality": {},
                "alerts": [],
                "failure_taxonomy": [],
            }

        job_ids = [metric.get("job_id") for metric in metrics_list if metric.get("job_id")]
        jobs_list = []
        if job_ids:
            jobs_list = await jobs.find({"job_id": {"$in": job_ids}}).to_list(None)

        averages = {
            "citation_rate": round(TelemetryCollector._sanitize_average(metrics_list, "citation_rate"), 3),
            "diversity_score": round(TelemetryCollector._sanitize_average(metrics_list, "diversity_score"), 3),
            "quality_score": round(TelemetryCollector._sanitize_average(metrics_list, "quality_score"), 3),
            "latency_ms": round(TelemetryCollector._sanitize_average(metrics_list, "latency_ms"), 3),
            "error_rate": round(TelemetryCollector._sanitize_average(metrics_list, "error_rate"), 3),
            "alternatives_count": round(TelemetryCollector._sanitize_average(metrics_list, "alternatives_count"), 3),
            "title_novelty_score": round(TelemetryCollector._sanitize_average(metrics_list, "title_novelty_score"), 3),
            "risk_specificity_score": round(TelemetryCollector._sanitize_average(metrics_list, "risk_specificity_score"), 3),
        }

        alerts = TelemetryCollector.evaluate_alerts(metrics_list, jobs_list=jobs_list)
        failure_counts = Counter(TelemetryCollector._classify_failure_reason(job) for job in jobs_list)

        latest = metrics_list[0]
        dashboard = {
            "total_jobs_sampled": len(metrics_list),
            "status": "ok",
            "quality": {
                "citation_rate": round(float(latest.get("citation_rate", 0) or 0), 3),
                "diversity_score": round(float(latest.get("diversity_score", 0) or 0), 3),
                "quality_score": round(float(latest.get("quality_score", 0) or 0), 3),
            },
            "grounding": {
                "citation_rate": round(averages["citation_rate"], 3),
                "error_rate": round(averages["error_rate"], 3),
                "alternatives_count": round(averages["alternatives_count"], 3),
                "title_novelty_score": round(averages["title_novelty_score"], 3),
                "risk_specificity_score": round(averages["risk_specificity_score"], 3),
            },
            "performance": {
                "latency_ms": round(averages["latency_ms"], 3),
            },
            "averages": averages,
            "alerts": alerts,
            "failure_taxonomy": [
                {"reason": reason, "count": count}
                for reason, count in failure_counts.most_common()
                if count > 0
            ],
            "latest_job_id": latest.get("job_id"),
            "latest_timestamp": latest.get("timestamp"),
        }

        return dashboard

    @staticmethod
    async def get_citation_stats() -> Dict[str, Any]:
        """Get citation-specific statistics from recent jobs."""
        db = await get_database()
        telemetry = db["telemetry_metrics"]

        metrics_list = (
            await telemetry.find().sort("timestamp", -1).limit(50).to_list(None)
        )

        if not metrics_list:
            return {"status": "No data", "average_citation_rate": 0}

        citation_rates = [m.get("citation_rate", 0) for m in metrics_list]
        avg_rate = statistics.mean(citation_rates) if citation_rates else 0

        return {
            "status": "ok",
            "average_citation_rate": round(avg_rate, 3),
            "sample_count": len(metrics_list),
            "min": round(min(citation_rates), 3) if citation_rates else 0,
            "max": round(max(citation_rates), 3) if citation_rates else 0,
        }
