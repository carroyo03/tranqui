import React, { useState } from 'react';
import { sendPasswordReset } from '../services/auth';

interface ForgotPasswordProps {
    onNext: () => void;
    onBack: () => void;
}

export const ForgotPassword: React.FC<ForgotPasswordProps> = ({ onNext, onBack }) => {
    const [email, setEmail] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        const { success, error } = await sendPasswordReset(email);
        setLoading(false);
        if (success) {
            onNext();
        } else {
            setError(error);
        }
    };

    return (
        <div className="flex flex-col h-full bg-white relative animate-fade-in">
            {/* Back Button */}
            <div className="absolute top-4 left-4 z-20">
                <button
                    onClick={onBack}
                    className="w-10 h-10 rounded-full flex items-center justify-center text-gray-500 hover:bg-gray-50 transition-colors"
                >
                    <span className="material-symbols-outlined">arrow_back</span>
                </button>
            </div>

            <div className="flex-1 flex flex-col px-8 pt-20">
                <div className="mb-8">
                    <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight mb-3">
                        Recuperar contraseña
                    </h1>
                    <p className="text-gray-500 text-lg leading-relaxed">
                        Introduce tu email y te enviaremos las instrucciones para restablecerla.
                    </p>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm font-medium animate-fade-in text-center">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-sm font-bold text-gray-900 block ml-1">Correo electrónico</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="hola@ejemplo.com"
                            required
                            className="w-full bg-gray-50 border-2 border-transparent focus:bg-white focus:border-primary/50 focus:ring-4 focus:ring-primary/10 rounded-2xl px-5 py-4 text-gray-900 placeholder-gray-400 transition-all outline-none font-medium text-lg"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-primary hover:bg-primary-hover text-white text-lg font-bold py-4 rounded-full shadow-lg shadow-primary/25 transition-all transform active:scale-[0.98] mt-4 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Enviando...' : 'Enviar enlace de recuperación'}
                    </button>
                </form>
            </div>
        </div>
    );
};