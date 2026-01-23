"use client";

import { useState, useCallback } from "react";
import { UploadCloud, FileText, Loader2, Type, ClipboardPaste } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";  
import { motion, AnimatePresence } from "framer-motion";

interface AnalysisInputProps {
  onFileUpload: (file: File) => Promise<void>;
  onTextAnalysis: (text: string) => Promise<void>;
  isAnalyzing: boolean;
}

export function AnalysisInput({ onFileUpload, onTextAnalysis, isAnalyzing }: AnalysisInputProps) {
  const [activeTab, setActiveTab] = useState("file");
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleFileSubmit = () => {
    if (file) onFileUpload(file);
  };

  const handleTextSubmit = () => {
    if (text.length >= 350) onTextAnalysis(text);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <Tabs defaultValue="file" className="w-full" onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2 mb-4">
          <TabsTrigger value="file">
             <FileText className="mr-2 h-4 w-4" />
             Upload Document
          </TabsTrigger>
          <TabsTrigger value="text">
             <Type className="mr-2 h-4 w-4" />
             Paste Text
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="file">
            <Card
                className={cn(
                "relative border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-all duration-200 min-h-[300px]",
                isDragging ? "border-primary bg-primary/5" : "border-border",
                "hover:border-primary/50 hover:bg-muted/50"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileChange}
                accept=".txt,.pdf,.docx"
                disabled={isAnalyzing}
                />
                
                <AnimatePresence mode="wait">
                {!file ? (
                    <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center gap-4"
                    >
                    <div className="p-4 rounded-full bg-muted">
                        <UploadCloud className="h-10 w-10 text-muted-foreground" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold">Upload Document</h3>
                        <p className="text-sm text-muted-foreground mt-1">
                        Drag & drop or <label htmlFor="file-upload" className="text-primary cursor-pointer hover:underline">browse</label>
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                        Supports PDF, DOCX, TXT
                        </p>
                    </div>
                    </motion.div>
                ) : (
                    <motion.div
                    key="filled"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    className="flex flex-col items-center gap-6"
                    >
                    <div className="p-4 rounded-full bg-primary/10">
                        <FileText className="h-10 w-10 text-primary" />
                    </div>
                    <div>
                        <h3 className="text-lg font-medium">{file.name}</h3>
                        <p className="text-sm text-muted-foreground">
                        {(file.size / 1024).toFixed(2)} KB
                        </p>
                    </div>
                    <div className="flex gap-3">
                        <label htmlFor="file-upload">
                            <Button variant="outline" size="sm" disabled={isAnalyzing}>Change File</Button>
                        </label>
                        <Button onClick={handleFileSubmit} disabled={isAnalyzing}>
                        {isAnalyzing ? (
                            <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analyzing...
                            </>
                        ) : (
                            "Analyze File"
                        )}
                        </Button>
                    </div>
                    </motion.div>
                )}
                </AnimatePresence>
            </Card>
        </TabsContent>
        
        <TabsContent value="text">
            <Card className="min-h-[300px] flex flex-col">
                <CardContent className="pt-6 flex-1 flex flex-col gap-4">
                    <Textarea 
                        placeholder="Paste your text here for analysis... (Minimum 350 characters)" 
                        className="flex-1 min-h-[200px] resize-none font-mono text-sm leading-relaxed focus:ring-0"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        disabled={isAnalyzing}
                    />
                    <div className="flex items-center justify-between gap-4">
                        <div className={cn(
                            "text-xs transition-colors",
                            text.length >= 350 ? "text-green-600 font-medium" : "text-muted-foreground"
                        )}>
                            {text.length} / 350 characters
                        </div>
                        <Button onClick={handleTextSubmit} disabled={isAnalyzing || text.length < 350} className="w-full sm:w-auto">
                            {isAnalyzing ? (
                                <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Analyzing...
                                </>
                            ) : (
                                <>
                                <ClipboardPaste className="mr-2 h-4 w-4" />
                                Analyze Text
                                </>
                            )}
                        </Button>
                    </div>
                </CardContent>
            </Card>
        </TabsContent>
      </Tabs>
      
    </div>
  );
}
