from pydantic import BaseModel, Field


class SegmentMetadata(BaseModel):
    segment_id: str = Field(description="Folder name / segment identifier")
    document_type: str = Field(
        description="Type of document (e.g. Letter, Work Order, Annexure)"
    )
    purpose: str = Field(description="Purpose or function of the document")
    parties: list[str] = Field(
        default_factory=list, description="Parties mentioned (URC, ISRO, etc.)"
    )
    dates: list[str] = Field(
        default_factory=list, description="Key dates found in the text"
    )
    reference_numbers: list[str] = Field(
        default_factory=list, description="Reference numbers, WO numbers, etc."
    )
    monetary_amounts: list[str] = Field(
        default_factory=list, description="Monetary amounts mentioned"
    )
    summary: str = Field(description="Concise summary of the document content")
