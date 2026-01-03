from app.models import SearchResult

def format_chromadb_results(docs, metas, dists) -> list[SearchResult]:
    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        results.append(
            SearchResult(
                file=meta.get("file"),
                start_page=meta.get("start_page"),
                end_page=meta.get("end_page"),
                header=meta.get("header"),
                header_type=meta.get("header_type"),
                part=meta.get("part"),
                chapter=meta.get("chapter"),
                section=meta.get("section"),
                subchunk_index=meta.get("subchunk_index"),
                article_index=meta.get("article_index"),
                similarity=round(dist, 4),
                content=doc.strip().replace("\n", " ")
            )
        )
    return results