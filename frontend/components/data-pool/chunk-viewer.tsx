"use client";

import { useEffect, useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getDocumentChunks } from "@/lib/api";
import { FileText, Loader2 } from "lucide-react";

interface ChunkViewerProps {
  filename: string;
}

interface ChunkData {
    id: number;
    metadata: {
        source: string;
        chunk_index: number;
        text_snippet: string; // Wait, snippet is truncated. Real functionality might need full text?
        // The backend `ingest` stores full chunks in FAISS but for metadata it stores snippet.
        // Let's check backend `plagiarism.py` add_to_index.
        // It stores `metadatas` which has `text_snippet`.
        // The actual vectors are in FAISS.
        // The *full text* of the chunk is NOT currently stored in metadata store in `ingest.py`!
        // It's only saving `chunk[:100]`.
        // I need to fix backend to store full text if I want to view it.
    }
}

// STOP: I realized I need to update backend to store full text in metadata if I want to display it here.
// Current backend implementation:
// metadatas.append({ "source": file.filename, "chunk_index": i, "text_snippet": chunk[:100] + "..." })
// I should add "text": chunk to this metadata.

export function ChunkViewer({ filename }: ChunkViewerProps) {
  const [chunks, setChunks] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (open) {
        setLoading(true);
        getDocumentChunks(filename)
            .then(setChunks)
            .finally(() => setLoading(false));
    }
  }, [open, filename]);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm">View Chunks</Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[540px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            {filename}
          </SheetTitle>
          <SheetDescription>
            Inspecting stored vectors and text segments.
          </SheetDescription>
        </SheetHeader>
        
        <div className="mt-6 h-full pb-10">
            {loading ? (
                <div className="flex justify-center py-10">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
            ) : (
                <ScrollArea className="h-[85vh] pr-4">
                    <div className="space-y-4">
                        {chunks.map((chunk) => (
                            <div key={chunk.id} className="p-4 rounded-lg border bg-card text-card-foreground shadow-sm">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-xs font-mono text-muted-foreground">ID: {chunk.id}</span>
                                    <span className="text-xs bg-muted px-2 py-1 rounded">Chunk #{chunk.metadata.chunk_index}</span>
                                </div>
                                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                    {chunk.metadata.text || chunk.metadata.text_snippet}
                                </p>
                            </div>
                        ))}
                    </div>
                </ScrollArea>
            )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
