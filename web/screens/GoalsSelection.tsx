import React, { useState } from 'react';
import { UserData } from '../types';

interface GoalsSelectionProps {
  onNext: (goals: string[]) => void;
  onBack: () => void;
  initialGoals: string[];
}

export const GoalsSelection: React.FC<GoalsSelectionProps> = ({ onNext, onBack, initialGoals }) => {
  const [selectedGoals, setSelectedGoals] = useState<string[]>(initialGoals);

  const goalsList = [
    { id: 'house', icon: 'home', label: 'Comprar una casa' },
    { id: 'emergency', icon: 'savings', label: 'Fondo de emergencia' },
    { id: 'travel', icon: 'flight', label: 'Viajar más' },
    { id: 'debt', icon: 'credit_card_off', label: 'Salir de deudas' },
    { id: 'invest', icon: 'trending_up', label: 'Invertir (Crypto/Bolsa)' },
    { id: 'peace', icon: 'self_improvement', label: 'Sentirme tranquilo/a' },
  ];

  const toggleGoal = (label: string) => {
    if (selectedGoals.includes(label)) {
      setSelectedGoals(selectedGoals.filter(g => g !== label));
    } else {
      setSelectedGoals([...selectedGoals, label]);
    }
  };

  return (
    <div className="flex flex-col h-full animate-fade-in relative px-8">
        <div className="mt-4 mb-8">
            <h1 className="text-3xl font-bold text-gray-900 tracking-tight mb-3">
                ¿Qué te importa?
            </h1>
            <p className="text-gray-500 text-lg font-medium">(Selecciona todos los que apliquen)</p>
        </div>

        <div className="flex-1 overflow-y-auto pb-32 space-y-4 -mx-2 px-2 scroll-smooth">
            {goalsList.map((goal) => {
                const isSelected = selectedGoals.includes(goal.label);
                return (
                    <label key={goal.id} className="cursor-pointer group relative w-full block">
                        <input 
                            type="checkbox" 
                            className="sr-only" 
                            checked={isSelected}
                            onChange={() => toggleGoal(goal.label)}
                        />
                        <div className={`flex items-center justify-between p-5 rounded-3xl border transition-all duration-300 ${
                            isSelected 
                                ? 'border-primary bg-purple-50 shadow-md transform scale-[1.02]' 
                                : 'border-gray-100 bg-white shadow-sm hover:border-primary/30 hover:shadow-md'
                        }`}>
                            <div className="flex items-center gap-4">
                                <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
                                    isSelected ? 'bg-primary text-white' : 'bg-gray-50 text-gray-400 group-hover:bg-primary/10 group-hover:text-primary'
                                }`}>
                                    <span className="material-symbols-outlined text-[24px]">{goal.icon}</span>
                                </div>
                                <span className={`text-lg font-semibold ${isSelected ? 'text-primary' : 'text-gray-700'}`}>
                                    {goal.label}
                                </span>
                            </div>
                            <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all ${
                                isSelected ? 'border-primary bg-primary' : 'border-gray-200 bg-transparent'
                            }`}>
                                <span className={`material-symbols-outlined text-white text-[14px] transition-transform ${
                                    isSelected ? 'scale-100' : 'scale-0'
                                }`}>
                                    check
                                </span>
                            </div>
                        </div>
                    </label>
                );
            })}
        </div>

        {/* Floating Bottom Button with Blur Fade */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white/95 to-transparent pt-12 pb-8 px-8 z-20">
            <button
                onClick={() => onNext(selectedGoals)}
                disabled={selectedGoals.length === 0}
                className={`w-full py-4 rounded-full text-lg font-bold shadow-lg transition-all duration-300 transform active:scale-[0.98] ${
                    selectedGoals.length > 0 
                    ? 'bg-primary text-white hover:bg-primary-hover shadow-primary/30 ring-4 ring-primary/10' 
                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                }`}
            >
                Continuar
            </button>
        </div>
    </div>
  );
};