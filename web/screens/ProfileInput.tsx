import React, { useState } from 'react';
import { UserData } from '../types';

interface ProfileInputProps {
  onNext: (data: Partial<UserData>) => void;
  onBack: () => void;
  initialData: Partial<UserData>;
}

export const ProfileInput: React.FC<ProfileInputProps> = ({ onNext, onBack, initialData }) => {
  const [age, setAge] = useState(initialData.age || 24);
  const [occupation, setOccupation] = useState(initialData.occupation || 'Trabajo a tiempo completo');
  const [income, setIncome] = useState(initialData.disposableIncome || '€300-500/mes');

  const handleNext = () => {
    onNext({
      age,
      occupation,
      disposableIncome: income
    });
  };

  return (
    <div className="flex flex-col h-full animate-fade-in px-8">
        <div className="mt-4 mb-8">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-3">
                Cuéntanos un poco sobre ti
            </h1>
            <p className="text-gray-500 text-lg font-medium">
                (Tarda 30 segundos)
            </p>
        </div>

        <div className="space-y-8 flex-1 overflow-y-auto pb-4 -mx-2 px-2">
            {/* Age Slider */}
            <div className="group">
                <div className="flex justify-between items-end mb-4 px-1">
                    <label className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Edad</label>
                    <span className="text-4xl font-bold text-primary">{age}</span>
                </div>
                <div className="bg-white rounded-[2rem] p-8 border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
                    <input
                        type="range"
                        min="18"
                        max="40"
                        value={age}
                        onChange={(e) => setAge(parseInt(e.target.value))}
                        className="w-full h-4 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-primary"
                        style={{
                            backgroundImage: `linear-gradient(to right, #5841D8 0%, #5841D8 ${(age - 18) / (40 - 18) * 100}%, #F3F4F6 ${(age - 18) / (40 - 18) * 100}%, #F3F4F6 100%)`
                        }}
                    />
                     <div className="flex justify-between text-xs font-bold text-gray-400 mt-4 px-1">
                        <span>18</span>
                        <span>40</span>
                    </div>
                </div>
            </div>

            {/* Occupation Select */}
            <div className="group">
                <label className="block text-sm font-semibold text-gray-700 mb-3 ml-1 uppercase tracking-wide">Actualmente...</label>
                <div className="relative">
                    <select 
                        value={occupation}
                        onChange={(e) => setOccupation(e.target.value)}
                        className="w-full bg-white border border-gray-200 text-gray-900 font-medium py-5 pl-6 pr-12 rounded-2xl focus:outline-none focus:ring-4 focus:ring-primary/10 focus:border-primary transition-all shadow-sm appearance-none cursor-pointer hover:border-primary/50 text-base"
                    >
                        <option>Trabajo a tiempo completo</option>
                        <option>Trabajo a tiempo parcial</option>
                        <option>Estudiante</option>
                        <option>Freelance / Autónomo</option>
                        <option>Buscando empleo</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-5 text-primary">
                        <span className="material-symbols-outlined">expand_more</span>
                    </div>
                </div>
            </div>

            {/* Income Select */}
            <div className="group">
                <label className="block text-sm font-semibold text-gray-700 mb-3 ml-1 uppercase tracking-wide">
                    Dinero libre al mes
                </label>
                <div className="relative">
                    <select
                        value={income}
                        onChange={(e) => setIncome(e.target.value)}
                        className="w-full bg-white border border-gray-200 text-gray-900 font-medium py-5 pl-6 pr-12 rounded-2xl focus:outline-none focus:ring-4 focus:ring-primary/10 focus:border-primary transition-all shadow-sm appearance-none cursor-pointer hover:border-primary/50 text-base"
                    >
                        <option>€0-100/mes</option>
                        <option>€100-300/mes</option>
                        <option>€300-500/mes</option>
                        <option>€500-1000/mes</option>
                        <option>€1000+/mes</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-5 text-primary">
                        <span className="material-symbols-outlined">expand_more</span>
                    </div>
                </div>
            </div>
        </div>

        <div className="mt-auto pt-4 pb-8">
             <button
                onClick={handleNext}
                className="w-full py-4 rounded-full bg-primary text-white text-lg font-bold hover:bg-primary-hover hover:scale-[1.02] active:scale-[0.98] transition-all shadow-lg shadow-primary/25 focus:outline-none ring-4 ring-primary/10"
            >
                Continuar
            </button>
        </div>
    </div>
  );
};