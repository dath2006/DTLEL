"use client";

import { useEffect, useState, useRef } from "react";
import { getDocuments, ingestDocument, DocumentInfo, clearDataPool, deleteDocument } from "@/lib/api";
import { DocumentTable } from "@/components/data-pool/document-table";
import { Button } from "@/components/ui/button";
import { PlusCircle, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function DataPoolPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const docs = await getDocuments();
      setDocuments(docs);
      setError(null);
    } catch (err: any) {
      setError("Failed to load documents. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIngesting(true);
    try {
      await ingestDocument(file);
      await fetchData(); // Refresh list
    } catch (err: any) {
      setError(err.message || "Failed to ingest document");
    } finally {
      setIngesting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // Reset input
      }
    }
  };

  const handleClearData = async () => {
    if (!confirm("Are you sure you want to clear ALL data? This cannot be undone.")) return;
    
    setClearing(true);
    try {
        await clearDataPool();
        await fetchData();
    } catch (err: any) {
        setError(err.message || "Failed to clear data pool");
    } finally {
        setClearing(false);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
        await deleteDocument(filename);
        await fetchData();
    } catch (err: any) {
        setError(err.message || "Failed to delete document");
    }
  };

  return (
    <main className="flex flex-1 flex-col gap-4 p-4 lg:gap-8 lg:p-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
             <h1 className="text-lg font-semibold md:text-2xl">Data Pool</h1>
             <p className="text-sm text-muted-foreground">
                Manage the reference documents used for Plagiarism Detection.
             </p>
        </div>
        <div className="flex gap-2">
            <Button variant="destructive" size="sm" onClick={handleClearData} disabled={clearing || loading}>
                {clearing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Trash2 className="mr-2 h-4 w-4" />}
                Clear All Data
            </Button>
            <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
            </Button>
            <Button size="sm" onClick={() => fileInputRef.current?.click()} disabled={ingesting}>
                {ingesting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <PlusCircle className="mr-2 h-4 w-4" />}
                Add Document
            </Button>
            <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept=".txt,.pdf,.docx"
                onChange={handleFileUpload}
            />
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
            <CardHeader className="py-4">
                <CardTitle className="text-sm font-medium">Total Sentences</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">
                    {documents.reduce((acc, doc) => acc + doc.chunk_count, 0)}
                </div>
            </CardContent>
        </Card>
      </div>

      <DocumentTable documents={documents} onDelete={handleDeleteDocument} />
    </main>
  );
}
