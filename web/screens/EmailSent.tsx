import React from 'react';

interface EmailSentProps {
  onBackToLogin: () => void;
}

export const EmailSent: React.FC<EmailSentProps> = ({ onBackToLogin }) => {
  return (
    <div className="flex flex-col h-full bg-white relative animate-fade-in items-center justify-center px-8 text-center">
        
        {/* Success Icon */}
        <div className="w-24 h-24 bg-purple-50 rounded-full flex items-center justify-center mb-8 relative">
            <div className="absolute inset-0 bg-primary/10 rounded-full animate-ping-slow"></div>
            <span className="material-symbols-outlined text-primary text-5xl">mark_email_read</span>
        </div>

        <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight mb-4">
            ¡Correo enviado!
        </h1>
        
        <p className="text-gray-500 text-lg leading-relaxed mb-10 max-w-xs mx-auto">
            Hemos enviado un código de verificación a tu correo. Por favor, revisa tu bandeja de entrada y la carpeta de spam.
        </p>

        <button 
            onClick={onBackToLogin}
            className="w-full bg-white text-primary border-2 border-primary hover:bg-purple-50 text-lg font-bold py-4 rounded-full shadow-sm transition-all transform active:scale-[0.98]"
        >
            Volver al inicio de sesión
        </button>
    </div>
  );
};