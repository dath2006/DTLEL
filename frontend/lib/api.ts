import { AnalysisReport, IngestResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

export async function uploadDocument(file: File): Promise<AnalysisReport> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Failed to analyze document");
  }

  return res.json();
}

export async function analyzeText(text: string): Promise<AnalysisReport> {
  const res = await fetch(`${BASE_URL}/analyze/text`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Failed to analyze text");
  }

  return res.json();
}

export async function ingestDocument(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/ingest`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Failed to ingest document");
  }

  return res.json();
}

export interface DocumentInfo {
    filename: string;
    chunk_count: number;
    upload_timestamp: string;
}

export async function getDocuments(): Promise<DocumentInfo[]> {
    const res = await fetch(`${BASE_URL}/documents`);
    if (!res.ok) {
        throw new Error("Failed to fetch documents");
    }
    return res.json();
}

export async function getDocumentChunks(filename: string): Promise<any[]> {
    const res = await fetch(`${BASE_URL}/documents/${filename}/chunks`);
    if (!res.ok) {
        throw new Error("Failed to fetch chunks");
    }
    return res.json();
}

export async function deleteDocument(filename: string): Promise<{ message: string; chunks_removed: number }> {
    const res = await fetch(`${BASE_URL}/documents/${filename}`, {
        method: "DELETE",
    });
    if (!res.ok) {
        throw new Error("Failed to delete document");
    }
    return res.json();
}

export async function clearDataPool(): Promise<{ message: string }> {
    const res = await fetch(`${BASE_URL}/documents/clear`, {
        method: "DELETE",
    });
    if (!res.ok) {
        throw new Error("Failed to clear data pool");
    }
    return res.json();
}

export async function checkHealth(): Promise<{ status: string }> {
  const HEALTH_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace('/api/v1', '') || "http://127.0.0.1:8000";
  const res = await fetch(`${HEALTH_URL}/health`);
  return res.json();
}
