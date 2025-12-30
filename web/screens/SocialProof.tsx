import React, { useEffect, useState } from 'react';
import { UserData } from '../types';

interface SocialProofProps {
  stressLevel: UserData['stressLevel'];
  onNext: () => void;
  onBack: () => void;
}

export const SocialProof: React.FC<SocialProofProps> = ({ stressLevel, onNext, onBack }) => {
  
  // Define data based on stress level
  let numericPercentage = 78;
  let text = 'de personas de tu edad sienten lo mismo.';
  
  if (stressLevel === 'low') {
    numericPercentage = 10;
    text = 'de personas logran mantener esa calma. ¡Qué top!';
  } else if (stressLevel === 'medium') {
    numericPercentage = 65;
    text = 'también tienen dudas de vez en cuando.';
  }

  // Animation state for the chart
  const [displayedPercentage, setDisplayedPercentage] = useState(0);

  useEffect(() => {
    // Animate the percentage value for the chart after mount
    const timer = setTimeout(() => {
        setDisplayedPercentage(numericPercentage);
    }, 100);
    return () => clearTimeout(timer);
  }, [numericPercentage]);

  // SVG Configuration
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (displayedPercentage / 100) * circumference;

  return (
    <div className="flex flex-col h-full relative z-0 animate-fade-in px-8">
        <div className="mt-4 mb-8">
            <h1 className="text-3xl font-bold text-gray-900 text-center tracking-tight">
                {stressLevel === 'low' ? '¡Increíble!' : 'No estás solo/a.'}
            </h1>
        </div>
        
        {/* Animated Main Card */}
        <div className="flex-1 flex items-center justify-center w-full mb-8">
            <div className="w-full bg-white/60 backdrop-blur-xl border-2 border-primary/10 rounded-[2.5rem] p-8 shadow-2xl shadow-primary/5 animate-float relative overflow-hidden group hover:border-primary/30 transition-colors duration-500 flex flex-col items-center justify-center">
                 
                 {/* Inner decorative gradient */}
                 <div className="absolute inset-0 bg-gradient-to-br from-white via-transparent to-purple-50/50 opacity-80"></div>
                 
                 {/* Sparkles decoration */}
                 <div className="absolute top-5 right-5 animate-pulse-glow z-20">
                    <span className="material-symbols-outlined text-primary/40 text-2xl">auto_awesome</span>
                 </div>

                 <div className="relative z-10 flex flex-col items-center text-center w-full">
                    
                    {/* Realistic Percentage Graphic (Donut Chart) */}
                    <div className="relative w-48 h-48 mb-8 flex items-center justify-center">
                        
                        {/* Background Glow */}
                        <div className="absolute inset-4 bg-primary/5 rounded-full blur-2xl"></div>

                        <svg className="w-full h-full -rotate-90 transform overflow-visible" viewBox="0 0 200 200">
                            {/* Track */}
                            <circle 
                                cx="100" cy="100" r={radius} 
                                fill="none" 
                                stroke="#F3F4F6" 
                                strokeWidth="20" 
                                strokeLinecap="round" 
                            />
                            {/* Progress Indicator */}
                            <circle 
                                cx="100" cy="100" r={radius} 
                                fill="none" 
                                stroke="url(#progressGradient)" 
                                strokeWidth="20" 
                                strokeDasharray={circumference} 
                                strokeDashoffset={strokeDashoffset} 
                                strokeLinecap="round"
                                className="transition-[stroke-dashoffset] duration-[1500ms] ease-out"
                            />
                            <defs>
                                <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stopColor="#A78BFA" /> {/* Lighter Purple */}
                                    <stop offset="100%" stopColor="#5841D8" /> {/* Primary Purple */}
                                </linearGradient>
                            </defs>
                        </svg>

                        {/* Floating Percentage Badge */}
                        <div className="absolute bottom-6 right-2 bg-white/90 backdrop-blur-sm px-4 py-2 rounded-2xl shadow-lg border border-gray-100 animate-bounce-slow flex items-center justify-center transform hover:scale-105 transition-transform">
                            <span className="text-primary font-extrabold text-xl">{numericPercentage}%</span>
                        </div>
                    </div>

                    <p className="text-gray-600 text-lg font-medium leading-relaxed max-w-[90%]">
                        {text}
                    </p>
                 </div>
            </div>
        </div>

        {/* Community Pill */}
        <div className="flex justify-center mb-10">
            <div className="bg-white/80 backdrop-blur-md px-6 py-4 rounded-full flex items-center gap-4 shadow-lg shadow-gray-200/50 border border-white/50 transform transition-transform hover:scale-105 duration-300">
                <div className="flex -space-x-3 shrink-0">
                    <div className="w-10 h-10 rounded-full border-2 border-white bg-indigo-100 flex items-center justify-center text-[10px] font-bold text-indigo-700 shadow-sm">JM</div>
                    <div className="w-10 h-10 rounded-full border-2 border-white bg-purple-100 flex items-center justify-center text-[10px] font-bold text-purple-700 shadow-sm">AK</div>
                    <div className="w-10 h-10 rounded-full border-2 border-white bg-pink-100 flex items-center justify-center text-[10px] font-bold text-pink-700 shadow-sm">L</div>
                </div>
                <div className="flex flex-col">
                     <span className="text-[10px] uppercase tracking-wider text-gray-400 font-bold">Comunidad Tranqui</span>
                     <p className="text-xs text-gray-600 leading-tight">
                        <span className="text-gray-900 font-bold">127.834</span> han encontrado paz.
                    </p>
                </div>
            </div>
        </div>

        <div className="w-full mt-auto pb-8">
             <button
                onClick={onNext}
                className="w-full py-4 rounded-full bg-gradient-to-r from-primary to-purple-600 text-white text-lg font-bold shadow-lg shadow-primary/25 hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] transition-all duration-300 focus:outline-none ring-4 ring-primary/10"
            >
                Continuar
            </button>
        </div>
        
        {/* Background elements */}
        <div className="absolute top-0 left-0 w-full h-full pointer-events-none -z-10 overflow-hidden">
             <div className="absolute top-[20%] left-[-20%] w-64 h-64 bg-purple-200/30 rounded-full blur-3xl animate-blob"></div>
             <div className="absolute bottom-[20%] right-[-20%] w-64 h-64 bg-blue-200/30 rounded-full blur-3xl animate-blob animation-delay-2000"></div>
        </div>
    </div>
  );
};