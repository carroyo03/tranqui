import React from 'react';
import { UserData } from '../types';

interface StressQuestionProps {
  onNext: (stressLevel: UserData['stressLevel']) => void;
  onBack: () => void;
}

export const StressQuestion: React.FC<StressQuestionProps> = ({ onNext, onBack }) => {
  const options: { label: string; value: UserData['stressLevel'] }[] = [
    { label: 'Sí, mucho', value: 'high' },
    { label: 'A veces', value: 'medium' },
    { label: 'En realidad no', value: 'low' },
  ];

  return (
    <div className="flex flex-col h-full animate-fade-in px-8 relative">
        {/* Background elements for depth */}
        <div className="absolute top-0 left-0 w-full h-1/2 bg-gradient-to-b from-gray-50 to-white -z-10" />

        <div className="flex-1 flex flex-col justify-center pb-20">
            <h1 className="text-3xl font-bold text-gray-900 mb-4 leading-tight tracking-tight">
                ¿Pensar en <span className="text-primary">dinero</span> <br />
                te estresa?
            </h1>
            <p className="text-gray-400 font-medium mb-12 text-lg">
                Sé sincer@, aquí no juzgamos.
            </p>
            
            <div className="flex flex-col gap-5 w-full">
                {options.map((option) => (
                    <button
                        key={option.value}
                        onClick={() => onNext(option.value)}
                        className="group relative w-full py-6 px-6 text-left bg-white border border-gray-100 rounded-3xl shadow-sm hover:border-primary/50 hover:shadow-glow hover:-translate-y-1 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-primary/10 active:scale-[0.98]"
                    >
                        <div className="flex items-center justify-between">
                            <span className="text-xl font-medium text-gray-700 group-hover:text-primary transition-colors">
                                {option.label}
                            </span>
                            <div className="w-6 h-6 rounded-full border-2 border-gray-200 group-hover:border-primary group-hover:bg-primary flex items-center justify-center transition-all duration-300">
                                <span className="material-symbols-outlined text-white text-[14px] opacity-0 group-hover:opacity-100 transition-opacity">
                                    check
                                </span>
                            </div>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    </div>
  );
};