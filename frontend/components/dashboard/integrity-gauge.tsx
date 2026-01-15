"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface IntegrityGaugeProps {
  score: number; // 0 to 1
}

export function IntegrityGauge({ score }: IntegrityGaugeProps) {
  const percentage = Math.round(score * 100);
  
  // Color logic
  let colorClass = "text-green-500";
  let statusText = "High Integrity";
  if (score < 0.8) {
    colorClass = "text-yellow-500";
    statusText = "Suspicious";
  }
  if (score < 0.5) {
    colorClass = "text-red-500";
    statusText = "Likely Fabricated";
  }

  return (
    <div className="flex flex-col items-center justify-center p-6">
      <div className="relative w-48 h-48 flex items-center justify-center">
        {/* Background Circle */}
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="96"
            cy="96"
            r="80"
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            className="text-muted/20"
          />
          {/* Progress Circle */}
          <motion.circle
            initial={{ pathLength: 0 }}
            animate={{ pathLength: score }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            cx="96"
            cy="96"
            r="80"
            stroke="currentColor"
            strokeWidth="12"
            fill="transparent"
            strokeLinecap="round"
            className={cn("drop-shadow-lg", colorClass)}
            style={{
                strokeDasharray: "502", // 2 * pi * 80
                strokeDashoffset: "0" // Handled by pathLength in framer-motion? No, framer-motion pathLength works on 0-1.
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className={cn("text-5xl font-bold tracking-tighter", colorClass)}
          >
            {percentage}%
          </motion.span>
          <span className="text-sm font-medium text-muted-foreground uppercase tracking-widest mt-1">Score</span>
        </div>
      </div>
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1 }}
        className="mt-4 text-center"
      >
        <h3 className={cn("text-xl font-semibold", colorClass)}>{statusText}</h3>
      </motion.div>
    </div>
  );
}
