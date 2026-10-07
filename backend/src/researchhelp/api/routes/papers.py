from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status

from researchhelp.api.deps import paper_service
from researchhelp.api.schemas import PaperOut, UploadedPaper, UploadResponse
from researchhelp.services.paper_service import PaperService

router = APIRouter(prefix="/papers", tags=["papers"])
Papers = Annotated[PaperService, Depends(paper_service)]


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_papers(
    background: BackgroundTasks,
    papers: Papers,
    files: Annotated[list[UploadFile], File(description="One or more PDF files")],
):
    """Upload PDFs. Each is stored, registered as ``processing`` and indexed in the background;
    poll ``GET /papers/{id}`` until its status is ``ready`` (or ``failed``)."""
    # Validate every file before storing any, so a bad file rejects the whole request cleanly.
    payloads = []
    for f in files:
        data = f.file.read(papers.max_upload_bytes + 1)
        papers.validate_upload(f.filename or "", data)
        payloads.append((f.filename or "upload.pdf", data))

    results = []
    for filename, data in payloads:
        record, needs_ingestion = papers.register_upload(filename, data)
        if needs_ingestion:
            background.add_task(papers.ingest, record.id)
        results.append(UploadedPaper(paper=PaperOut.of(record), duplicate=not needs_ingestion))
    return UploadResponse(papers=results)


@router.get("", response_model=list[PaperOut])
def list_papers(papers: Papers):
    return [PaperOut.of(r) for r in papers.list_all()]


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str, papers: Papers):
    return PaperOut.of(papers.get(paper_id))


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_paper(paper_id: str, papers: Papers):
    """Remove the paper's chunks from the index, its file, and its record."""
    papers.delete(paper_id)
