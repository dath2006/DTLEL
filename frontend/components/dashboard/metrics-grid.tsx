import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AnalysisMetrics } from "@/lib/types";
import { Fingerprint, BarChart3, Binary, BookOpen, MessageSquare, AlertTriangle, Layers } from "lucide-react";

interface MetricsGridProps {
  metrics: AnalysisMetrics;
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  const baseItems = [
    {
      title: "AI Detection",
      value: `${(metrics.ai_score * 100).toFixed(0)}%`,
      desc: `${metrics.ensemble_method === 'ensemble_avg' ? 'Ensemble' : 'RoBERTa'} model`,
      icon: Binary,
      color: metrics.ai_score > 0.5 ? "text-red-500" : "text-green-500"
    },
    {
      title: "Plagiarism",
      value: `${(metrics.plagiarism_score * 100).toFixed(0)}%`,
      desc: `${metrics.plagiarism_percentage.toFixed(1)}% text matched`,
      icon: BookOpen,
      color: metrics.plagiarism_score > 0.3 ? "text-red-500" : "text-green-500"
    },
    {
      title: "AI Phrases",
      value: `${(metrics.stylometry_score * 100).toFixed(0)}%`,
      desc: `${metrics.ai_phrase_count} phrases found`,
      icon: MessageSquare,
      color: metrics.stylometry_score > 0.3 ? "text-orange-500" : "text-green-500"
    },
    {
      title: "Burstiness",
      value: metrics.burstiness_score.toFixed(2),
      desc: metrics.burstiness_score > 0.5 ? "Human-like variation" : "Uniform (AI-like)",
      icon: BarChart3,
      color: metrics.burstiness_score > 0.5 ? "text-green-500" : "text-amber-500"
    }
  ];

  // Add new metrics if available
  const newItems = [];
  
  if (metrics.stylometry_score !== null && metrics.stylometry_score !== undefined) {
    newItems.push({
      title: "AI Phrases",
      value: `${(metrics.stylometry_score * 100).toFixed(0)}%`,
      desc: `${metrics.ai_phrase_count || 0} phrases detected`,
      icon: MessageSquare,
      color: "text-orange-500"
    });
  }
  
  if (metrics.sentence_count !== null && metrics.sentence_count !== undefined) {
    newItems.push({
      title: "AI Sentences",
      value: `${metrics.ai_sentence_count || 0}/${metrics.sentence_count}`,
      desc: "Flagged / Total sentences",
      icon: Layers,
      color: "text-cyan-500"
    });
  }

  if (metrics.perplexity_flux !== undefined) {
    newItems.push({
      title: "Perplexity Flux",
      value: metrics.perplexity_flux.toFixed(2),
      desc: metrics.perplexity_flux > 0.4 ? "High variation" : "Suspiciously Uniform",
      icon: AlertTriangle,
      color: metrics.perplexity_flux > 0.4 ? "text-green-500" : "text-amber-500"
    });
  }

  // Remove duplicates by title (since baseItems has AI Phrases)
  const uniqueItems = new Map();
  [...baseItems, ...newItems].forEach(item => {
    // Only add if not already present or if it's an overwrite (AI Phrases handled poorly before)
    if (!uniqueItems.has(item.title) || item.title === "AI Phrases") {
      uniqueItems.set(item.title, item);
    }
  });

  const items = Array.from(uniqueItems.values());

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
      {items.map((item) => (
        <Card key={item.title}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {item.title}
            </CardTitle>
            <item.icon className={`h-4 w-4 ${item.color}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{item.value}</div>
            <p className="text-xs text-muted-foreground">
              {item.desc}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

