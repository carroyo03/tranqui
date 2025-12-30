import React from 'react';

interface ProgressBarProps {
  currentStep: number;
  totalSteps: number;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ currentStep, totalSteps }) => {
  // We map steps to visual dots. 
  // Step 1 corresponds to index 0 visually, etc.
  const dots = Array.from({ length: totalSteps }, (_, i) => i + 1);

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-semibold text-gray-400 tracking-wide">
        {currentStep}/{totalSteps}
      </span>
      <div className="flex gap-1.5">
        {dots.map((stepNum) => {
           const isActive = stepNum <= currentStep;
           const isCurrent = stepNum === currentStep;
           
           return (
            <div
                key={stepNum}
                className={`h-2 rounded-full transition-all duration-300 ${
                    isCurrent ? 'w-6 bg-primary' : 'w-2'
                } ${
                    !isCurrent && isActive ? 'bg-primary/40' : ''
                } ${
                    !isActive ? 'bg-gray-200' : ''
                }`}
            />
           );
        })}
      </div>
    </div>
  );
};
