import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { DocumentInfo } from "@/lib/api";
import { FileText, Trash2 } from "lucide-react";
import { ChunkViewer } from "./chunk-viewer";
import { Button } from "@/components/ui/button";

interface DocumentTableProps {
  documents: DocumentInfo[];
  onDelete: (filename: string) => void;
}

export function DocumentTable({ documents, onDelete }: DocumentTableProps) {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Document Name</TableHead>
            <TableHead>Sentences</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Action</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {documents.length === 0 ? (
            <TableRow>
              <TableCell colSpan={4} className="h-24 text-center">
                No documents in the pool.
              </TableCell>
            </TableRow>
          ) : (
            documents.map((doc) => (
              <TableRow key={doc.filename}>
                <TableCell className="font-medium flex items-center gap-2">
                    <FileText className="h-4 w-4 text-blue-500" />
                    {doc.filename}
                </TableCell>
                <TableCell>{doc.chunk_count}</TableCell>
                <TableCell>
                    <Badge variant="secondary" className="bg-green-100 text-green-800 hover:bg-green-100">Indexed</Badge>
                </TableCell>
                <TableCell className="text-right flex justify-end gap-2">
                    <ChunkViewer filename={doc.filename} />
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive hover:text-destructive/90" onClick={() => onDelete(doc.filename)}>
                        <Trash2 className="h-4 w-4" />
                    </Button>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
