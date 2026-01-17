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

from app.utils.logger import append_log, record_event
from app.database.connection import get_database
from app.database.vector_store import upsert_chunk, query_similar_chunks


class ContextBuilder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.use_web = os.getenv('DGS_USE_WEB', '0') == '1'

    async def search_candidates(self, query: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """Return candidate source metadata. By default returns simulated candidates.

        If `DGS_USE_WEB=1` the code attempts a lightweight DuckDuckGo HTML search.
        """
        record_event(level='INFO', action='deep_rag.search.start', message=f"search_candidates for: {query}")
        candidates: List[Dict[str, Any]] = []
        if self.use_web:
            try:
                params = {'q': query}
                async with httpx.AsyncClient(timeout=20.0) as client:
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
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
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

    async def build_knowledge_base(self, query: str, top_k: int = 5):
        """Full Deep RAG ingestion flow: search -> filter -> scrape -> chunk -> embed -> upsert."""
        record_event(level='INFO', action='deep_rag.implementation.start', message=f"Begin Deep RAG ingestion for query: {query}")

        candidates = await self.search_candidates(query, top_n=15)
        filtered = self.filter_candidates(candidates, keep=7)
        scraped = await self.parallel_scrape(filtered, concurrency=5)

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
                    'created_at': datetime.utcnow(),
                    'ttl_days': 30
                }
                await upsert_chunk(chunk_doc)
                inserted_ids.append(chunk_doc['id'])

        record_event(level='INFO', action='deep_rag.implementation.done', message=f"Deep RAG ingestion completed", details={"query": query, "candidates": len(candidates), "filtered": len(filtered), "scraped": len(scraped), "total_chunks": total_chunks, "inserted_count": len(inserted_ids)})
        return {'status': 'ok', 'query': query, 'inserted_ids': inserted_ids, 'total_chunks': total_chunks}

    async def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # fallback to vector search
        docs = await query_similar_chunks(query=query, k=k)
        return docs

    async def get_context_for_reasoner(self, query: str, k: int = 5, min_confidence: float = 0.0):
        chunks = await self.retrieve_relevant_chunks(query, k=k)
        context_confidence = 0.0
        if chunks:
            sims = [c.get('_similarity_score', 0.0) for c in chunks]
            context_confidence = max(sims) if sims else 0.0
        return {"chunks": chunks, "context_confidence": context_confidence}
