"use client";

import { useState } from "react";
import { Header } from "@/components/dashboard/header";
import { AnalysisInput } from "@/components/dashboard/analysis-input";
import { SeparateScores } from "@/components/dashboard/separate-scores";
import { MetricsGrid } from "@/components/dashboard/metrics-grid";
import { TextHeatmap } from "@/components/dashboard/text-heatmap";
import { StylometryCard } from "@/components/dashboard/stylometry-card";
import { AnalysisVerdict } from "@/components/dashboard/analysis-verdict";
import { AnalysisReport } from "@/lib/types";
import { uploadDocument, analyzeText } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function AnalysisPage() {
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await uploadDocument(file);
      setReport(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTextAnalysis = async (text: string) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await analyzeText(text);
      setReport(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setReport(null);
    setError(null);
  };

  return (
    <div className="flex flex-col h-full w-full">
      <Header />
      
      <main className="flex-1 w-full max-w-7xl mx-auto p-6 md:p-12">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {!report ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center min-h-[60vh] space-y-8"
          >
            <div className="text-center space-y-2">
              <h2 className="text-4xl font-extrabold tracking-tight lg:text-5xl">
                Verify Content Integrity
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Advanced forensic analysis using Hybrid AI detection, Stylometry, and Semantic Plagiarism search.
              </p>
            </div>
            
            <AnalysisInput 
                onFileUpload={handleUpload} 
                onTextAnalysis={handleTextAnalysis}
                isAnalyzing={isAnalyzing} 
            />
            
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-8"
          >
            <div className="flex items-center justify-between">
              <Button variant="ghost" onClick={resetAnalysis} className="pl-0 hover:bg-transparent hover:text-primary">
                <ChevronLeft className="mr-2 h-4 w-4" />
                New Analysis
              </Button>
              <div className="text-sm text-muted-foreground">
                Report ID: <span className="font-mono">{report.report_id.split('-')[0]}</span>
              </div>
            </div>

            {/* NEW: Executive Summary / Verdict Card */}
            <AnalysisVerdict report={report} />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
              {/* Left Col: Separate Scores & Metrics */}
              <div className="lg:col-span-1 space-y-6">
                 {/* Three Separate Score Gauges */}
                 <SeparateScores 
                   ai_score={report.ai_score}
                   plagiarism_score={report.plagiarism_score}
                   stylometry_score={report.stylometry_score}
                   ensemble_method={report.metrics.ensemble_method}
                 />
                 <MetricsGrid metrics={report.metrics} />
                 
                 {/* Stylometry Card */}
                 {report.stylometry && (
                   <StylometryCard stylometry={report.stylometry} />
                 )}
              </div>

              {/* Right Col: Heatmap with Tabs */}
              <div className="lg:col-span-2">
                <div className="bg-card rounded-xl border shadow-sm">
                   {/* Header for text view */}
                   <div className="px-4 py-3 border-b flex items-center justify-between bg-muted/30">
                      <h3 className="font-semibold">Analysis View</h3>
                      <div className="flex gap-2 text-xs">
                        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500"></span>Plagiarism</span>
                        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500"></span>AI Generated</span>
                        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500"></span>AI Phrases</span>
                      </div>
                   </div>
                   
                   <Tabs defaultValue="chunks" className="w-full">
                     <TabsList className="w-full justify-start px-4 bg-transparent border-b rounded-none">
                       <TabsTrigger value="chunks">Chunk View</TabsTrigger>
                       {report.sentence_scores && report.sentence_scores.length > 0 && (
                         <TabsTrigger value="sentences">Sentence View ({report.sentence_scores.length})</TabsTrigger>
                       )}
                     </TabsList>
                     
                     <TabsContent value="chunks" className="m-0">
                       <TextHeatmap segments={report.segments} />
                     </TabsContent>
                     
                     {report.sentence_scores && report.sentence_scores.length > 0 && (
                       <TabsContent value="sentences" className="m-0 p-4">
                         <div className="space-y-2 max-h-[500px] overflow-y-auto">
                           {report.sentence_scores.map((sentence, i) => (
                             <div 
                               key={i}
                               className={`group relative p-4 rounded-lg border transition-all hover:shadow-md ${
                                 sentence.is_ai_generated 
                                   ? 'bg-purple-500/10 border-purple-500/30' 
                                   : 'bg-background hover:bg-muted/50'
                               }`}
                             >
                               <div className="flex flex-col gap-3">
                                 {/* Top Row: Text and Score Badge */}
                                 <div className="flex justify-between items-start gap-4">
                                   <p className="text-sm leading-relaxed">{sentence.text}</p>
                                   <div className={`px-2 py-1 rounded text-xs font-bold shrink-0 ${
                                     sentence.is_ai_generated 
                                       ? 'bg-purple-500/20 text-purple-900 dark:text-purple-500' 
                                       : 'bg-green-500/20 text-green-900 dark:text-green-500'
                                   }`}>
                                     {(sentence.ai_probability * 100).toFixed(0)}% AI
                                   </div>
                                 </div>

                                 {/* Bottom Row: Insights (Perplexity) */}
                                 <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t border-dashed">
                                     <div className="flex items-center gap-1.5" title="Lower perplexity = predictable = likely AI">
                                       <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                                       <span>Predictability (PPL): <span className="font-mono font-medium text-foreground">{sentence.perplexity?.toFixed(1) || 'N/A'}</span></span>
                                     </div>
                                     
                                     {/* Interpretation Badge */}
                                     {sentence.perplexity && sentence.perplexity < 25 && (
                                       <span className="px-1.5 py-0.5 rounded-sm bg-amber-500/10 text-amber-600 text-[10px] uppercase font-bold tracking-wider">
                                         High Predictability
                                       </span>
                                     )}
                                     {sentence.perplexity && sentence.perplexity > 80 && (
                                       <span className="px-1.5 py-0.5 rounded-sm bg-blue-500/10 text-blue-600 text-[10px] uppercase font-bold tracking-wider">
                                         High Complexity (Human)
                                       </span>
                                     )}
                                     
                                     {sentence.window_count > 1 && (
                                       <span className="ml-auto text-[10px] opacity-60">
                                         Verified in {sentence.window_count} passes
                                       </span>
                                     )}
                                 </div>
                               </div>
                             </div>
                           ))}
                         </div>
                       </TabsContent>
                     )}
                   </Tabs>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}

