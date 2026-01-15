import { AnalysisReport } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle2, AlertTriangle, Fingerprint, BookOpen, AlertOctagon } from "lucide-react";
import { cn } from "@/lib/utils";

interface AnalysisVerdictProps {
  report: AnalysisReport;
}

export function AnalysisVerdict({ report }: AnalysisVerdictProps) {
  const { ai_score, plagiarism_score, stylometry_score } = report;

  // --- Logic for Verdict ---
  let status: "safe" | "warning" | "danger" | "inconclusive" = "safe";
  let title = "Content Appears Authentic";
  let description = "Our analysis found no significant evidence of AI generation or plagiarism.";
  let highlights: string[] = [];

  // 1. Critical: BOTH High (Mixed Integrity Issues)
  if (plagiarism_score > 0.20 && ai_score > 0.65) {
    status = "danger";
    title = "Multiple Integrity Issues Detected";
    description = "Critical Alert: This content exhibits strong evidence of both plagiarism AND AI generation.";
    
    highlights.push(`Found ${(plagiarism_score * 100).toFixed(0)}% matching content from external sources.`);
    highlights.push(`High probability of AI generation (${(ai_score * 100).toFixed(0)}%).`);
    highlights.push("Content appears to be a hybrid of copied text and AI-synthesized material.");
  }

  // 2. High Plagiarism Only (Legal/Academic Risk)
  else if (plagiarism_score > 0.20) {
    status = "danger";
    title = "Significant Plagiarism Detected";
    description = "This content contains substantial amounts of text matching external sources. Immediate review is recommended.";
    highlights.push(`Found ${(plagiarism_score * 100).toFixed(0)}% matching content from external sources.`);
  } 
  
  // 3. High AI Only
  else if (ai_score > 0.65) {
    status = "warning"; // Warning for AI, unless extremely high
    if (ai_score > 0.85) status = "danger";
    
    title = "Likely AI-Generated Content";
    description = "The text exhibits strong statistical patterns typical of AI language models.";
    highlights.push(`AI probability is high at ${(ai_score * 100).toFixed(0)}%.`);
    
    if (stylometry_score > 0.6) {
        highlights.push("Stylometric analysis confirms repetitive, machine-like phrasing.");
    }
  }

  // 4. Mixed / Moderate (The "Generous" Middle Ground)
  else if (ai_score > 0.4 || plagiarism_score > 0.1) {
    status = "warning";
    title = "Potential Integrity Issues";
    description = "Some signals suggest the content may not be entirely original or human-written, though results are not definitive.";
    
    if (plagiarism_score > 0.1) highlights.push(`Minor plagiarism detected (${(plagiarism_score * 100).toFixed(0)}%).`);
    if (ai_score > 0.4) highlights.push(`Moderate AI probability (${(ai_score * 100).toFixed(0)}%).`);
  }

  // --- Styles ---
  const colorMap = {
    safe: "bg-green-500/10 border-green-500/20 text-green-700 dark:text-green-400",
    warning: "bg-amber-500/10 border-amber-500/20 text-amber-700 dark:text-amber-400",
    danger: "bg-red-500/10 border-red-500/20 text-red-700 dark:text-red-400",
    inconclusive: "bg-blue-500/10 border-blue-500/20 text-blue-700 dark:text-blue-400",
  };

  const iconMap = {
    safe: CheckCircle2,
    warning: AlertTriangle,
    danger: AlertOctagon, // or Siren
    inconclusive: Fingerprint,
  };

  const Icon = iconMap[status];

  return (
    <Card className={cn("border-l-4 shadow-sm", colorMap[status].replace("bg-", "border-l-").split(" ")[0])}>
      <CardHeader className="pb-2 flex flex-row items-center gap-4 space-y-0">
        <div className={cn("p-2 rounded-full", colorMap[status])}>
          <Icon className="w-6 h-6" />
        </div>
        <div className="space-y-1">
          <CardTitle className="text-xl">{title}</CardTitle>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </CardHeader>
      
      {highlights.length > 0 && (
        <CardContent>
          <div className="mt-2 p-3 bg-muted/50 rounded-md text-sm">
             <span className="font-semibold mb-2 block text-foreground">Key Findings:</span>
             <ul className="list-disc list-inside space-y-1 text-muted-foreground">
               {highlights.map((point, i) => (
                 <li key={i}>{point}</li>
               ))}
             </ul>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
