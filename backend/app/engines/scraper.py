"""ContextBuilder (Deep RAG) implementation.

Implements a local-friendly Deep RAG pipeline with safe fallbacks when web
search or heavy scraping deps are unavailable. Steps:
  1) search_candidates (Top-15) [simulated by default]
  2) filter_candidates (~5-7)
  3) parallel_scrape -> fetch page HTML
  4) clean & chunk -> 400-700 char chunks
  5) embed & upsert chunks

Each major step records structured events via `record_event`/`append_log`.
"""
from typing import List, Dict, Any, Optional
import uuid
import os
import re
import asyncio
from datetime import datetime
import httpx
from html import unescape
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import pytz

from backend.app.utils.logger import append_log, record_event
from backend.app.database.connection import get_database
from backend.app.database.vector_store import upsert_chunk, query_similar_chunks, ensure_text_index
from backend.app.utils.metrics import metrics, track_latency


class ContextBuilder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.use_web = os.getenv('DGS_USE_WEB', '0') == '1'
        self.respect_robots = os.getenv('DGS_RESPECT_ROBOTS', '1') == '1'
        self.scraper_timeout_seconds = float(os.getenv('SCRAPER_TIMEOUT_SECONDS', '8'))
        self.parallel_scrape_max_workers = int(
            os.getenv('SCRAPER_CONCURRENCY', os.getenv('PARALLEL_SCRAPE_MAX_WORKERS', '4'))
        )
        self.scraper_ttl_days = int(os.getenv('SCRAPER_TTL_DAYS', '30'))
        self.rag_candidates_top_n = int(os.getenv('RAG_CANDIDATES_TOP_N', '12'))
        self.rag_filtered_keep = int(os.getenv('RAG_FILTERED_KEEP', '5'))
        self._robots_cache: Dict[str, RobotFileParser] = {}

    async def _is_scrape_allowed(self, url: str) -> bool:
        if not url or url.startswith('http://sim.test/'):
            return True

        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            return False

        if not self.respect_robots:
            return True

        cache_key = f"{parsed.scheme}://{parsed.netloc}".lower()
        parser = self._robots_cache.get(cache_key)
        if parser is None:
            robots_url = f"{cache_key}/robots.txt"
            try:
                timeout = min(self.scraper_timeout_seconds, 5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(robots_url)
                parser = RobotFileParser()
                parser.set_url(robots_url)
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser.parse([])
                self._robots_cache[cache_key] = parser
            except Exception as e:
                append_log(f"robots_check_failed {robots_url}: {e}")
                return False

        try:
            return bool(parser.can_fetch('*', url))
        except Exception:
            return False

    async def search_candidates(self, query: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """Return candidate source metadata. By default returns simulated candidates.

        If `DGS_USE_WEB=1` the code attempts a lightweight DuckDuckGo HTML search.
        """
        record_event(level='INFO', action='deep_rag.search.start', message=f"search_candidates for: {query}")
        candidates: List[Dict[str, Any]] = []
        if self.use_web:
            try:
                params = {'q': query}
                async with httpx.AsyncClient(timeout=self.scraper_timeout_seconds) as client:
                    resp = await client.get('https://html.duckduckgo.com/html/', params=params)
                    text = resp.text
                    # crude link/snippet extraction
                    links = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"', text)
                    snippets = re.findall(r'<a[^>]+class="result__a"[^>]*>([^<]+)</a>', text)
                    for i, url in enumerate(links[:top_n]):
                        title = snippets[i] if i < len(snippets) else url
                        candidates.append({'title': title.strip(), 'url': url, 'snippet': title.strip()})
            except Exception as e:
                append_log(f"search_candidates web search failed: {e}")
                # fall through to simulated

        if not candidates:
            # Simulated candidate list (safe offline fallback)
            for i in range(min(top_n, 15)):
                candidates.append({
                    'title': f"Simulated Source {i+1} for {query}",
                    'url': f"http://sim.test/{i+1}",
                    'snippet': f"Simulated snippet for {query} from source {i+1}"
                })

        record_event(level='INFO', action='deep_rag.search.done', message=f"found {len(candidates)} candidates", details={'query': query, 'count': len(candidates)})
        return candidates

    def filter_candidates(self, candidates: List[Dict[str, Any]], keep: int = 7) -> List[Dict[str, Any]]:
        """Apply simple filtering heuristics; currently deterministic picking of first `keep` items."""
        record_event(level='INFO', action='deep_rag.filter.start', message=f"filtering {len(candidates)} candidates")
        # TODO: implement domain whitelist/blacklist and recency checks
        filtered = candidates[:keep]
        record_event(level='INFO', action='deep_rag.filter.done', message=f"filtered -> {len(filtered)} candidates")
        return filtered

    async def parallel_scrape(self, candidates: List[Dict[str, Any]], concurrency: int = 5) -> List[Dict[str, Any]]:
        """Fetch candidate pages concurrently and return cleaned text per source."""
        record_event(level='INFO', action='deep_rag.scrape.start', message=f"scraping {len(candidates)} sources")

        async def fetch(url: str) -> Optional[str]:
            if url.startswith('http://sim.test/'):
                return None
            if not await self._is_scrape_allowed(url):
                append_log(f"parallel_scrape: skipped disallowed url {url}")
                return None
            try:
                async with httpx.AsyncClient(timeout=self.scraper_timeout_seconds) as client:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return r.text
            except Exception as e:
                append_log(f"parallel_scrape: fetch failed {url}: {e}")
            return None

        sem = asyncio.Semaphore(concurrency)

        async def worker(c):
            async with sem:
                html = await fetch(c.get('url'))
                text = self.clean_html(html) if html else c.get('snippet', '')
                return {'title': c.get('title'), 'url': c.get('url'), 'text': text}

        tasks = [asyncio.create_task(worker(c)) for c in candidates]
        results = await asyncio.gather(*tasks)
        record_event(level='INFO', action='deep_rag.scrape.done', message=f"scraped {len(results)} sources")
        return results

    def clean_html(self, html: str) -> str:
        if not html:
            return ''
        # remove scripts/styles
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html, flags=re.S|re.I)
        # remove all tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = unescape(text)
        # collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_text(self, text: str, min_size: int = 400, max_size: int = 700) -> List[str]:
        """Chunk text into ~400-700 char segments attempting paragraph boundaries."""
        if not text:
            return []
        paragraphs = [p.strip() for p in re.split(r'\n{2,}|\r{2,}', text) if p.strip()]
        chunks: List[str] = []
        buffer = ''
        for p in paragraphs:
            if not buffer:
                buffer = p
            elif len(buffer) + len(p) + 1 <= max_size:
                buffer = buffer + '\n\n' + p
            else:
                if len(buffer) < min_size:
                    # try to extend with this paragraph anyway
                    buffer = buffer + '\n\n' + p
                    chunks.append(buffer[:max_size])
                    buffer = ''
                else:
                    chunks.append(buffer)
                    buffer = p
        if buffer:
            chunks.append(buffer)

        # ensure chunk sizes within bounds by further splitting if needed
        final: List[str] = []
        for c in chunks:
            if len(c) <= max_size:
                final.append(c)
            else:
                # split into max_size pieces
                for i in range(0, len(c), max_size):
                    final.append(c[i:i+max_size])
        return final

    async def build_knowledge_base(self, query: str, top_k: int = 5, session_id: Optional[str] = None, scraping_enabled: bool = True):
        """Full Deep RAG ingestion flow: search -> filter -> scrape -> chunk -> embed -> upsert.
        
        Per project guide: "provide a toggle to opt out of web scraping if user prefers privacy/legal safety"
        
        Args:
            query: Search query for knowledge base
            top_k: Number of top candidates to retrieve
            session_id: Optional session ID for per-session scraping control
            scraping_enabled: Whether to perform web scraping (per-session override)
        """
        with track_latency('rag.ingestion'):
            record_event(level='INFO', action='deep_rag.implementation.start', message=f"Begin Deep RAG ingestion for query: {query}", details={'scraping_enabled': scraping_enabled, 'session_id': session_id})
            
            candidates = await self.search_candidates(query, top_n=self.rag_candidates_top_n)
            filtered = self.filter_candidates(candidates, keep=self.rag_filtered_keep)
            
            # Check session-level scraping control (project guide requirement)
            if scraping_enabled and self.use_web:
                scraped = await self.parallel_scrape(filtered, concurrency=self.parallel_scrape_max_workers)
            else:
                # If scraping disabled, use only snippets (no actual web fetching)
                record_event(level='INFO', action='deep_rag.scraping_disabled', message=f"Scraping disabled for session {session_id}, using cached data only")
                scraped = [
                    {'title': c.get('title'), 'url': c.get('url'), 'text': c.get('snippet', '')}
                    for c in filtered
                ]
            
            inserted_ids: List[str] = []
            total_chunks = 0
            for src_idx, src in enumerate(scraped):
                content = src.get('text', '')
                chunks = self.chunk_text(content)
                total_chunks += len(chunks)
                for idx, ctext in enumerate(chunks):
                    chunk_doc = {
                        'id': str(uuid.uuid4()),
                        'content': ctext,
                        'source_url': src.get('url') or f"http://sim.test/{src_idx+1}",
                        'source_title': src.get('title'),
                        'chunk_index': idx,
                        'verification_status': 'unverified',
                        'created_at': datetime.now(pytz.UTC),
                        'ttl_days': self.scraper_ttl_days
                    }
                    await upsert_chunk(chunk_doc)
                    inserted_ids.append(chunk_doc['id'])

            # Log ingestion metrics
            metrics.record_metric(
                operation='rag.parallel_scrape',
                chunk_count=total_chunks,
                success=True,
                details={'sources_scraped': len(scraped), 'scraping_enabled': scraping_enabled}
            )

            record_event(level='INFO', action='deep_rag.implementation.done', message="Deep RAG ingestion completed", details={"query": query, "candidates": len(candidates), "filtered": len(filtered), "scraped": len(scraped), "total_chunks": total_chunks, "inserted_count": len(inserted_ids)})
            return {'status': 'ok', 'query': query, 'inserted_ids': inserted_ids, 'total_chunks': total_chunks}

    async def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using vector search with hybrid sparse fallback."""
        with track_latency('rag.retrieval'):
            # Vector search
            docs = await query_similar_chunks(query=query, k=k)
            
            cache_hit = len(docs) > 0
            top_similarity = max([d.get('_similarity_score', 0.0) for d in docs], default=0.0)
            
            # Log retrieval metrics
            metrics.record_metric(
                operation='rag.vector_search',
                cache_hit=cache_hit,
                similarity_score=top_similarity,
                chunk_count=len(docs),
                success=True
            )
            
            # Fallback: if no results or all low scores, use keyword search
            if not docs or all(d.get('_similarity_score', 0.0) < 0.7 for d in docs):
                record_event(level='INFO', action='hybrid_sparse_fallback', message=f"Fallback triggered for query: {query}")
                
                with track_latency('rag.text_fallback'):
                    # Perform BM25-like keyword search as fallback
                    db = await get_database()
                    coll = db['global_context']
                    await ensure_text_index()
                    keyword_results = await coll.find({
                        '$text': {'$search': query}
                    }, {
                        'score': {'$meta': 'textScore'}
                    }).sort('score', -1).limit(k).to_list(length=k)

                    for result in keyword_results:
                        text_score = float(result.get('score') or 0.0)
                        # Mongo textScore is not normalized to 0..1. Treat any
                        # keyword hit as a grounded fallback while preserving a
                        # little ordering from the raw text score.
                        result['_similarity_score'] = min(0.95, max(0.82, 0.80 + (text_score * 0.03)))
                        result['_retrieval_mode'] = 'text_fallback'
                    
                    fallback_count = len(keyword_results) if keyword_results else 0
                    metrics.record_metric(
                        operation='rag.text_search',
                        cache_hit=fallback_count > 0,
                        chunk_count=fallback_count,
                        success=fallback_count > 0,
                        details={'trigger': 'low_vector_similarity', 'threshold': 0.7}
                    )
                    
                    record_event(level='INFO', action='fallback_results', message=f"Fallback returned {fallback_count} chunks")
                    return keyword_results
            
            return docs

    async def get_context_for_reasoner(self, query: str, k: int = 5, min_confidence: float = 0.0):
        chunks = await self.retrieve_relevant_chunks(query, k=k)
        context_confidence = 0.0
        if chunks:
            sims = []
            for c in chunks:
                if '_similarity_score' not in c and 'score' in c:
                    text_score = float(c.get('score') or 0.0)
                    c['_similarity_score'] = min(0.95, max(0.82, 0.80 + (text_score * 0.03)))
                sims.append(float(c.get('_similarity_score', 0.0)))
            context_confidence = max(sims) if sims else 0.0
        return {"chunks": chunks, "context_confidence": context_confidence}
