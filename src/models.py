from typing import Optional

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


class TaxonomyNode(BaseModel):
    key: str = Field(description="Unique identifier for this category")
    name: str = Field(description="Human-readable category name")
    description: Optional[str] = Field(
        default=None, description="Brief description of what documents fit here"
    )
    children: list["TaxonomyNode"] = Field(
        default_factory=list, description="Nested subcategories"
    )


class ClusterAssignment(BaseModel):
    cluster_key: str = Field(description="Key from the taxonomy this cluster maps to")
    cluster_name: str = Field(description="Human-readable cluster name")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    is_primary: bool = Field(
        default=False, description="Whether this is the primary classification"
    )


class TypeClassification(BaseModel):
    segment_id: str
    clusters: list[ClusterAssignment] = Field(
        description="1-3 cluster assignments, one should have is_primary=true"
    )
