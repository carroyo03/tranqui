import React, { useState } from 'react';
import { registerWithEmail, loginWithGoogle } from '../services/auth';

interface SignupProps {
    onSignupSuccess: () => void;
    onGoToLogin: () => void;
    onBack: () => void;
}

export const Signup: React.FC<SignupProps> = ({ onSignupSuccess, onGoToLogin, onBack }) => {
    const [showPassword, setShowPassword] = useState(false);
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSignup = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        const { user, error } = await registerWithEmail(email, password);
        setLoading(false);
        if (user) {
            // Optionally update user profile with name
            onSignupSuccess();
        } else {
            setError(error);
        }
    };

    const handleGoogleSignup = async () => {
        setLoading(true);
        const { user, error } = await loginWithGoogle();
        setLoading(false);
        if (user) {
            onSignupSuccess();
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

            <div className="flex-1 flex flex-col px-8 pt-12 overflow-y-auto">
                {/* Header */}
                <div className="text-center mb-10 mt-8">
                    <h1 className="text-6xl md:text-7xl lg:text-8xl font-black tracking-tighter text-gray-900 drop-shadow-sm cursor-default leading-none">
                        Tranqui<span className="text-primary">.</span>
                    </h1>
                </div>

                {/* Social Buttons (Signup variant) */}
                <div className="space-y-4 mb-10">

                    <button onClick={handleGoogleSignup} disabled={loading} className="w-full bg-white text-gray-900 border border-gray-200 py-4 rounded-full font-medium flex items-center justify-center gap-3 hover:bg-gray-50 transition-colors active:scale-[0.98] disabled:opacity-70">
                        <svg className="w-5 h-5" viewBox="0 0 24 24">
                            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                        </svg>
                        {loading ? 'Cargando...' : 'Registrarse con Google'}
                    </button>
                </div>

                {/* Divider */}
                <div className="relative flex items-center justify-center mb-10">
                    <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-200"></div>
                    </div>
                    <div className="relative bg-white px-4 text-xs text-gray-400 font-bold uppercase tracking-wider">
                        o regístrate con email
                    </div>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm font-medium animate-fade-in text-center">
                        {error}
                    </div>
                )}

                {/* Form */}
                <form onSubmit={handleSignup} className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-sm font-bold text-gray-900 block ml-1">Nombre</label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Tu nombre"
                            className="w-full bg-gray-50 border-2 border-transparent focus:bg-white focus:border-primary/50 focus:ring-4 focus:ring-primary/10 rounded-2xl px-5 py-4 text-gray-900 placeholder-gray-400 transition-all outline-none font-medium text-lg"
                            required
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-bold text-gray-900 block ml-1">Correo electrónico</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="hola@ejemplo.com"
                            className="w-full bg-gray-50 border-2 border-transparent focus:bg-white focus:border-primary/50 focus:ring-4 focus:ring-primary/10 rounded-2xl px-5 py-4 text-gray-900 placeholder-gray-400 transition-all outline-none font-medium text-lg"
                            required
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-bold text-gray-900 block ml-1">Contraseña</label>
                        <div className="relative">
                            <input
                                type={showPassword ? "text" : "password"}
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                className="w-full bg-gray-50 border-2 border-transparent focus:bg-white focus:border-primary/50 focus:ring-4 focus:ring-primary/10 rounded-2xl px-5 py-4 text-gray-900 placeholder-gray-400 transition-all outline-none font-medium pr-12 tracking-widest text-lg"
                                required
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-primary transition-colors"
                            >
                                <span className="material-symbols-outlined text-[24px]">
                                    {showPassword ? 'visibility_off' : 'visibility'}
                                </span>
                            </button>
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-primary hover:bg-primary-hover text-white text-lg font-bold py-4 rounded-full shadow-lg shadow-primary/25 transition-all transform active:scale-[0.98] mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Creando cuenta...' : 'Crear cuenta'}
                    </button>
                </form>

                <div className="mt-auto py-10 text-center">
                    <p className="text-gray-500 text-sm font-medium">
                        ¿Ya tienes cuenta? <button onClick={onGoToLogin} className="text-primary font-bold hover:underline">Iniciar sesión</button>
                    </p>
                </div>
            </div>
        </div>
    );
};