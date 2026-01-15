"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Binary, BookOpen, MessageSquare } from "lucide-react";

interface ScoreGaugeProps {
  score: number;
  label: string;
  icon: React.ReactNode;
  description: string;
  inverted?: boolean; // If true, higher score = worse (default for AI detection)
}

function ScoreGauge({ score, label, icon, description, inverted = true }: ScoreGaugeProps) {
  const percentage = Math.round(score * 100);
  
  // Color logic (inverted: higher = red, lower = green)
  let colorClass = "text-green-500";
  let bgClass = "bg-green-500";
  if (inverted) {
    if (score > 0.3) {
      colorClass = "text-yellow-500";
      bgClass = "bg-yellow-500";
    }
    if (score > 0.6) {
      colorClass = "text-red-500";
      bgClass = "bg-red-500";
    }
  } else {
    if (score < 0.5) {
      colorClass = "text-yellow-500";
      bgClass = "bg-yellow-500";
    }
    if (score < 0.3) {
      colorClass = "text-red-500";
      bgClass = "bg-red-500";
    }
  }

  return (
    <div className="flex flex-col items-center p-4">
      <div className="flex items-center gap-2 mb-2">
        <span className={colorClass}>{icon}</span>
        <span className="text-sm font-medium text-muted-foreground">{label}</span>
      </div>
      <motion.div 
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className={cn("text-4xl font-bold", colorClass)}
      >
        {percentage}%
      </motion.div>
      <div className="w-full h-2 bg-muted rounded-full mt-2 overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={cn("h-full rounded-full", bgClass)}
        />
      </div>
      <p className="text-xs text-muted-foreground mt-2 text-center">{description}</p>
    </div>
  );
}

interface SeparateScoresProps {
  ai_score: number;
  plagiarism_score: number;
  stylometry_score: number;
  ensemble_method?: string;
}

export function SeparateScores({ ai_score, plagiarism_score, stylometry_score, ensemble_method }: SeparateScoresProps) {
  const aiLabel = ensemble_method === 'ensemble_avg' ? 'Ensemble' : 'RoBERTa';
  
  return (
    <div className="grid grid-cols-3 gap-2 p-4 bg-card rounded-xl border shadow-sm">
      <ScoreGauge 
        score={ai_score}
        label="AI Detection"
        icon={<Binary className="h-5 w-5" />}
        description={`${aiLabel} model`}
        inverted={true}
      />
      <ScoreGauge 
        score={plagiarism_score}
        label="Plagiarism"
        icon={<BookOpen className="h-5 w-5" />}
        description="Text matched"
        inverted={true}
      />
      <ScoreGauge 
        score={stylometry_score}
        label="AI Phrases"
        icon={<MessageSquare className="h-5 w-5" />}
        description="Phrase density"
        inverted={true}
      />
    </div>
  );
}
