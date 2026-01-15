import { AnalysisSegment } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AlertCircle, BookOpen, Fingerprint } from "lucide-react";

interface TextHeatmapProps {
  segments: AnalysisSegment[];
}

export function TextHeatmap({ segments }: TextHeatmapProps) {
  return (
    <ScrollArea className="h-[500px] w-full rounded-md border p-6 bg-card text-card-foreground leading-relaxed whitespace-pre-wrap">
      {segments.map((seg, i) => {
        let bgClass = "bg-transparent transition-colors hover:bg-muted";
        let textClass = "text-foreground";
        
        // Priority Logic: Plagiarism > AI (usually)
        if (seg.is_plagiarized) {
          if (seg.plagiarism_score > 0.9) {
             bgClass = "bg-red-500/30 hover:bg-red-500/40 border-b-2 border-red-500/50";
          } else {
             bgClass = "bg-yellow-500/30 hover:bg-yellow-500/40 border-b-2 border-yellow-500/50";
          }
        } else if (seg.ai_probability > 0.8) {
             bgClass = "bg-purple-500/30 hover:bg-purple-500/40 border-b-2 border-purple-500/50";
        } else if (seg.ai_probability > 0.5) {
             bgClass = "bg-purple-500/10 hover:bg-purple-500/20";
        }

        return (
          <Tooltip key={i} delayDuration={0}>
            <TooltipTrigger asChild>
              <span 
                className={cn("px-0.5 py-0.5 rounded cursor-help inline", bgClass, textClass)}
              >
                {seg.text}
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-md p-0 overflow-hidden bg-card border shadow-xl z-50">
               <div className="flex flex-col">
                  {/* Header Area */}
                  <div className={cn(
                    "px-3 py-2 text-xs font-semibold flex items-center justify-between border-b",
                     seg.is_plagiarized ? "bg-red-500/10 text-red-700 dark:text-red-400" : 
                     seg.ai_probability > 0.8 ? "bg-purple-500/10 text-purple-700 dark:text-purple-400" :
                     "bg-muted text-muted-foreground"
                  )}>
                     <span className="flex items-center gap-1.5">
                       {seg.is_plagiarized ? (
                         <><BookOpen className="w-3 h-3"/> Plagiarism Detected</>
                       ) : seg.ai_probability > 0.8 ? (
                         <><Fingerprint className="w-3 h-3"/> AI Generated Content</>
                       ) : "Content Details"}
                     </span>
                     <span>
                        {seg.is_plagiarized 
                          ? `${(seg.plagiarism_score * 100).toFixed(0)}% Match` 
                          : seg.ai_probability > 0.5 
                            ? `${(seg.ai_probability * 100).toFixed(0)}% AI` 
                            : "Original"}
                     </span>
                  </div>

                  {/* Body Area */}
                  <div className="p-3 text-xs space-y-3 bg-card text-card-foreground">
                     {/* Plagiarism Details */}
                     {seg.is_plagiarized && seg.source_metadata && (
                       <div className="space-y-2">
                          <div className="grid grid-cols-[80px_1fr] gap-2">
                              <span className="text-muted-foreground">Source:</span>
                              <span className="font-medium break-all">{seg.source_metadata.source}</span>
                              
                              <span className="text-muted-foreground">Similarity:</span>
                              <span className="font-medium">{(seg.plagiarism_score * 100).toFixed(1)}%</span>

                              {seg.source_metadata.start_char !== undefined && (
                                <>
                                  <span className="text-muted-foreground">Location:</span>
                                  <span className="font-mono text-[10px] text-muted-foreground">
                                    Chars {seg.source_metadata.start_char} - {seg.source_metadata.end_char}
                                  </span>
                                </>
                              )}
                          </div>
                          
                          {seg.source_metadata.text_snippet && (
                            <div className="mt-2 text-[10px] text-muted-foreground bg-muted/50 p-2 rounded border">
                               <p className="font-semibold mb-1 text-foreground/80">Source Text Match:</p>
                               <p className="italic font-serif leading-snug">"{seg.source_metadata.text_snippet}"</p>
                            </div>
                          )}
                       </div>
                     )}

                     {/* Fallback if no specific detailed metadata but flagged */}
                     {seg.is_plagiarized && !seg.source_metadata && (
                       <div className="flex items-center gap-2 text-amber-600 dark:text-amber-500">
                         <AlertCircle className="w-4 h-4" />
                         <span>Source metadata unavailable for this match.</span>
                       </div>
                     )}
                     
                     {/* AI Details if not plagiarized (or if desired to show both) */}
                     {!seg.is_plagiarized && (
                        <div className="space-y-1">
                           <div className="flex justify-between">
                              <span className="text-muted-foreground">AI Probability:</span>
                              <span className="font-mono font-bold">{(seg.ai_probability * 100).toFixed(1)}%</span>
                           </div>
                           <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-purple-500 transition-all" 
                                style={{ width: `${seg.ai_probability * 100}%` }} 
                              />
                           </div>
                        </div>
                     )}
                  </div>
               </div>
            </TooltipContent>
          </Tooltip>
        );
      })}
    </ScrollArea>
  );
}
