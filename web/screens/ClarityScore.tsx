import React, { useEffect, useState } from 'react';
import { ClarityInsight } from '../types';

interface ClarityScoreProps {
  insight: ClarityInsight;
  onRestart: () => void;
  onChat: () => void;
}

export const ClarityScore: React.FC<ClarityScoreProps> = ({ insight, onRestart, onChat }) => {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    // Animate the score number and ring
    const timer = setTimeout(() => {
        setAnimatedScore(insight.score);
    }, 300);
    return () => clearTimeout(timer);
  }, [insight.score]);

  // SVG Calculation
  const radius = 58; // Increased size
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  return (
    <div className="flex flex-col h-full animate-fade-in relative">
        {/* Subtle Background Gradient */}
        <div className="absolute top-0 left-0 right-0 h-64 bg-gradient-to-b from-purple-50/50 to-transparent -z-10" />

        <div className="flex-1 overflow-y-auto px-8 pb-32 pt-6 no-scrollbar">
            
            {/* Header Section */}
            <div className="flex flex-col items-center text-center mb-10">
                <span className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-6 bg-white/50 px-3 py-1 rounded-full backdrop-blur-sm border border-gray-100">
                    Análisis Completado
                </span>

                {/* Main Score Indicator */}
                <div className="relative w-48 h-48 mb-6 flex items-center justify-center">
                    {/* Outer Glow */}
                    <div className="absolute inset-0 bg-primary/5 rounded-full blur-2xl transform scale-110"></div>
                    
                    {/* SVG Ring */}
                    <svg className="w-full h-full -rotate-90 transform overflow-visible" viewBox="0 0 140 140">
                        {/* Background Track */}
                        <circle 
                            cx="70" cy="70" r={radius} 
                            fill="none" 
                            stroke="#F3F4F6" 
                            strokeWidth="8" 
                            strokeLinecap="round" 
                        />
                        {/* Animated Value */}
                        <circle 
                            cx="70" cy="70" r={radius} 
                            fill="none" 
                            stroke="url(#scoreGradient)" 
                            strokeWidth="8" 
                            strokeDasharray={circumference} 
                            strokeDashoffset={strokeDashoffset} 
                            strokeLinecap="round"
                            className="transition-[stroke-dashoffset] duration-[2000ms] cubic-bezier(0.2, 0.8, 0.2, 1)"
                        />
                        <defs>
                            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#A78BFA" />
                                <stop offset="100%" stopColor="#5841D8" />
                            </linearGradient>
                        </defs>
                    </svg>

                    {/* Central Number */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-br from-primary to-purple-600 tracking-tighter">
                            {animatedScore}
                        </span>
                        <span className="text-gray-300 text-xs font-bold mt-1">/ 100</span>
                    </div>
                </div>

                <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
                    Tu Puntuación de Claridad
                </h1>
            </div>

            {/* Insights Grid */}
            <div className="space-y-4 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
                
                {/* 1. Comparison Card (Green/Success Theme) */}
                <div className="bg-white p-5 rounded-3xl border border-gray-100 shadow-[0_2px_12px_-4px_rgba(0,0,0,0.05)] flex gap-4 items-start hover:border-emerald-200 transition-colors duration-300">
                    <div className="w-10 h-10 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0">
                        <span className="material-symbols-outlined text-[20px]">trending_up</span>
                    </div>
                    <div>
                        <h3 className="text-sm font-bold text-gray-900 mb-1">Tu posición</h3>
                        <p className="text-sm text-gray-600 leading-relaxed">
                            Estás por delante del <span className="font-bold text-emerald-600 bg-emerald-50 px-1 rounded">{insight.percentile}%</span> de personas con perfil similar.
                        </p>
                    </div>
                </div>

                {/* 2. Improvement Card (Blue/Info Theme) */}
                <div className="bg-white p-5 rounded-3xl border border-gray-100 shadow-[0_2px_12px_-4px_rgba(0,0,0,0.05)] flex gap-4 items-start hover:border-blue-200 transition-colors duration-300">
                    <div className="w-10 h-10 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                        <span className="material-symbols-outlined text-[20px]">lightbulb</span>
                    </div>
                    <div>
                        <h3 className="text-sm font-bold text-gray-900 mb-1">Observación</h3>
                        <p className="text-sm text-gray-600 leading-relaxed">
                            {insight.improvementText}
                        </p>
                    </div>
                </div>

                {/* 3. Action Card (Purple/Primary Theme) */}
                <div className="bg-purple-50/50 p-5 rounded-3xl border border-purple-100 flex gap-4 items-start relative overflow-hidden group">
                     {/* Decorative bg shape */}
                    <div className="absolute -right-4 -top-4 w-20 h-20 bg-purple-100/50 rounded-full blur-xl group-hover:bg-purple-200/50 transition-colors"></div>
                    
                    <div className="w-10 h-10 rounded-2xl bg-white text-primary flex items-center justify-center shrink-0 shadow-sm z-10">
                        <span className="material-symbols-outlined text-[20px]">bolt</span>
                    </div>
                    <div className="z-10">
                        <h3 className="text-sm font-bold text-primary mb-1">TranquiTip</h3>
                        <p className="text-sm text-gray-700 font-medium italic leading-relaxed">
                            "{insight.shortAdvice}"
                        </p>
                    </div>
                </div>
            </div>
        </div>

        {/* Sticky Bottom Actions */}
        <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-white via-white/95 to-transparent z-20">
            {/* UX Friendly Button Redesign */}
            <button
                onClick={onChat}
                className="w-full bg-primary hover:bg-primary-hover text-white py-4 px-8 rounded-[2rem] shadow-xl shadow-primary/25 flex items-center justify-between transition-all transform active:scale-[0.98] group border border-primary/20 relative overflow-hidden"
            >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
                
                <div className="flex flex-col items-start relative z-10">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-indigo-200 mb-0.5">Siguiente paso</span>
                    <span className="text-lg font-bold tracking-tight">Hablar con TranquiCoach</span>
                </div>
                
                <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm group-hover:bg-white/30 transition-colors border border-white/10 relative z-10">
                    <span className="material-symbols-outlined text-2xl">chat</span>
                </div>
            </button>
            
            <button
                onClick={onRestart}
                className="w-full mt-5 text-gray-400 text-xs font-bold hover:text-gray-600 transition-colors uppercase tracking-widest"
            >
                Empezar de nuevo
            </button>
        </div>
    </div>
  );
};