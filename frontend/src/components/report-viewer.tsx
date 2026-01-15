import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertCircle, FileText } from "lucide-react";

interface Segment {
    text: string;
    ai_detection: {
        is_ai: boolean;
        confidence: number;
        perplexity: number;
        reasons: string[];
    };
    plagiarism_detection: {
        is_plagiarized: boolean;
        score: number;
        matches: any[];
    };
}

interface ReportViewerProps {
    segments: Segment[];
}

export function ReportViewer({ segments }: ReportViewerProps) {
    return (
        <Card className="mt-6">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Detailed Analysis
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="prose max-w-none text-lg leading-relaxed p-4 border rounded-md bg-white min-h-[200px]">
                    <TooltipProvider>
                        {segments.map((segment, index) => {
                            const isAI = segment.ai_detection.is_ai;
                            const isPlag = segment.plagiarism_detection.is_plagiarized;

                            let bgClass = "";
                            let borderClass = "";
                            if (isAI && isPlag) {
                                bgClass = "bg-orange-100 hover:bg-orange-200";
                                borderClass = "border-b-2 border-orange-500";
                            } else if (isAI) {
                                bgClass = "bg-red-100 hover:bg-red-200";
                                borderClass = "border-b-2 border-red-500";
                            } else if (isPlag) {
                                bgClass = "bg-yellow-100 hover:bg-yellow-200";
                                borderClass = "border-b-2 border-yellow-500";
                            }

                            return (
                                <Tooltip key={index}>
                                    <TooltipTrigger asChild>
                                        <span 
                                            className={`inline transition-colors duration-200 px-1 rounded-sm ${bgClass} ${borderClass} cursor-default`}
                                        >
                                            {segment.text}{" "}
                                        </span>
                                    </TooltipTrigger>
                                    <TooltipContent className="max-w-sm">
                                        <div className="space-y-2">
                                            {isAI && (
                                                <div>
                                                    <p className="font-bold text-red-600 flex items-center gap-1">
                                                        <AlertCircle className="h-4 w-4" /> AI Detected
                                                    </p>
                                                    <p className="text-xs">Confidence: {segment.ai_detection.confidence.toFixed(1)}%</p>
                                                    {segment.ai_detection.reasons.length > 0 && (
                                                        <ul className="list-disc pl-4 text-xs text-muted-foreground">
                                                            {segment.ai_detection.reasons.map((r, i) => (
                                                                <li key={i}>{r}</li>
                                                            ))}
                                                        </ul>
                                                    )}
                                                </div>
                                            )}
                                            {isPlag && (
                                                <div className={isAI ? "mt-2 pt-2 border-t" : ""}>
                                                    <p className="font-bold text-yellow-600 flex items-center gap-1">
                                                        <AlertCircle className="h-4 w-4" /> Plagiarism Detected
                                                    </p>
                                                    <p className="text-xs">Score: {(segment.plagiarism_detection.score * 100).toFixed(1)}%</p>
                                                </div>
                                            )}
                                            {!isAI && !isPlag && (
                                                <p className="text-green-600 font-medium">Original Content</p>
                                            )}
                                        </div>
                                    </TooltipContent>
                                </Tooltip>
                            );
                        })}
                    </TooltipProvider>
                </div>
                
                <div className="mt-4 flex gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                        <span className="w-4 h-4 bg-red-100 border-b-2 border-red-500 rounded-sm"></span>
                        <span>AI Content</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="w-4 h-4 bg-yellow-100 border-b-2 border-yellow-500 rounded-sm"></span>
                        <span>Plagiarized Content</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="w-4 h-4 bg-orange-100 border-b-2 border-orange-500 rounded-sm"></span>
                        <span>Both</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
