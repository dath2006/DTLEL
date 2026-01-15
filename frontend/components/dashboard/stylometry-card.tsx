"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { StylemetryReport } from "@/lib/types";
import { MessageSquareWarning, Hash, TrendingUp } from "lucide-react";

interface StylometryCardProps {
  stylometry: StylemetryReport;
}

const categoryColors: Record<string, string> = {
  transitions: "bg-blue-500",
  abstract_nouns: "bg-purple-500",
  verbs: "bg-green-500",
  adjectives: "bg-amber-500",
  filler_phrases: "bg-red-500",
};

const categoryLabels: Record<string, string> = {
  transitions: "Transitions",
  abstract_nouns: "Abstract Nouns",
  verbs: "AI Verbs",
  adjectives: "AI Adjectives",
  filler_phrases: "Filler Phrases",
};

export function StylometryCard({ stylometry }: StylometryCardProps) {
  const scorePercent = Math.round(stylometry.stylometry_score * 100);
  
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <MessageSquareWarning className="h-5 w-5 text-amber-500" />
              AI Phrase Analysis
            </CardTitle>
            <CardDescription>
              Detected {stylometry.total_ai_phrases} AI-characteristic phrases
            </CardDescription>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-amber-500">{scorePercent}%</div>
            <div className="text-xs text-muted-foreground">Phrase Density</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">AI Phrase Saturation</span>
            <span className={scorePercent > 50 ? "text-red-500" : "text-green-500"}>
              {scorePercent > 70 ? "High" : scorePercent > 30 ? "Medium" : "Low"}
            </span>
          </div>
          <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all ${
                scorePercent > 70 ? 'bg-red-500' : scorePercent > 30 ? 'bg-amber-500' : 'bg-green-500'
              }`}
              style={{ width: `${scorePercent}%` }}
            />
          </div>
        </div>

        {/* Category breakdown */}
        <div className="space-y-2">
          <div className="text-sm font-medium flex items-center gap-1">
            <Hash className="h-4 w-4" />
            Category Breakdown
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stylometry.category_breakdown).map(([category, count]) => {
              const colorMap: Record<string, string> = {
                transitions: "#3b82f6",
                abstract_nouns: "#a855f7",
                verbs: "#22c55e",
                adjectives: "#f59e0b",
                filler_phrases: "#ef4444",
              };
              return (
                <Badge 
                  key={category} 
                  variant="secondary"
                  className="flex items-center gap-1.5"
                >
                  <span 
                    className="w-2 h-2 rounded-full" 
                    style={{ backgroundColor: colorMap[category] || "#6b7280" }}
                  />
                  {categoryLabels[category] || category}: {count}
                </Badge>
              );
            })}
          </div>
        </div>

        {/* Top phrases */}
        {stylometry.top_phrases.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium flex items-center gap-1">
              <TrendingUp className="h-4 w-4" />
              Most Frequent AI Phrases
            </div>
            <div className="flex flex-wrap gap-1">
              {stylometry.top_phrases.slice(0, 8).map(([phrase, count]) => (
                <Badge key={phrase} variant="outline" className="text-xs">
                  "{phrase}" × {count}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2 pt-2 border-t text-center">
          <div>
            <div className="text-lg font-semibold">{stylometry.total_ai_phrases}</div>
            <div className="text-xs text-muted-foreground">Total Phrases</div>
          </div>
          <div>
            <div className="text-lg font-semibold">{stylometry.unique_phrases}</div>
            <div className="text-xs text-muted-foreground">Unique</div>
          </div>
          <div>
            <div className="text-lg font-semibold">{stylometry.phrase_density.toFixed(1)}%</div>
            <div className="text-xs text-muted-foreground">Density</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
