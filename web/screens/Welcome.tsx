import React from 'react';

// New Leaf Logo Component
const LeafLogo = () => (
  <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-xl">
    <circle cx="50" cy="50" r="50" className="fill-white" />
    <g transform="translate(22, 22) scale(0.56)">
      <path d="M50 0C50 0 20 15 20 55C20 85 45 95 50 100C55 95 80 85 80 55C80 15 50 0 50 0Z" fill="#34D399" />
      <path d="M50 10V100" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 35L25 50" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 55L25 70" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 75L35 85" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 35L75 50" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 55L75 70" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 75L65 85" stroke="white" strokeWidth="4" strokeLinecap="round" />
    </g>
  </svg>
);

interface WelcomeProps {
  onNext: () => void;
  onLogin: () => void;
}

export const Welcome: React.FC<WelcomeProps> = ({ onNext, onLogin }) => {
  return (
    <div className="flex-1 flex flex-col md:flex-row items-center justify-center relative overflow-hidden h-full w-full">

      {/* Background - Full Screen on Mobile, Right Side on Desktop */}
      {/* Background - Full Screen on Mobile, Right Side on Desktop */}
      <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none z-0 bg-white">
        {/* Left Side Blobs */}
        <div className="absolute top-[-10%] left-[-10%] w-[40rem] h-[40rem] bg-indigo-50/50 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob"></div>
        <div className="absolute top-[20%] right-[40%] w-[30rem] h-[30rem] bg-blue-50/50 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob animation-delay-2000"></div>

        {/* Right Side Brand Gradient (Angled Split) */}
        <div className="hidden md:block absolute top-0 right-0 w-[50%] h-full bg-[#5841D8]">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 to-violet-600 opacity-90"></div>
          {/* Abstract Shapes on Dark BG */}
          <div className="absolute top-[-20%] right-[-10%] w-[40rem] h-[40rem] bg-white/10 rounded-full blur-3xl opacity-40 animate-pulse-glow"></div>
          <div className="absolute bottom-[-10%] left-[-20%] w-[30rem] h-[30rem] bg-indigo-400/20 rounded-full blur-3xl opacity-40"></div>
          <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
        </div>
      </div>

      {/* Main Container - Responsive Grid */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-6 md:px-12 lg:px-24 h-full flex flex-col md:flex-row md:items-center justify-between pointer-events-none md:pointer-events-auto">

        {/* Left Column: Content */}
        <div className="flex-1 flex flex-col items-center md:items-start justify-center h-full pt-10 md:pt-0 pointer-events-auto z-20">

          {/* Logo Container */}
          <div className="relative w-24 h-24 mb-8 animate-float flex items-center justify-center select-none">
            <div className="absolute inset-0 bg-emerald-100/50 rounded-full blur-xl animate-pulse-glow"></div>
            <div className="relative z-10 w-full h-full shadow-sm rounded-full">
              <LeafLogo />
            </div>
          </div>

          <div className="text-center md:text-left space-y-6 animate-fade-in-up opacity-0 max-w-lg" style={{ animationDelay: '0.2s', animationFillMode: 'forwards' }}>
            <h1 className="text-6xl md:text-7xl lg:text-8xl font-black tracking-tighter text-gray-900 drop-shadow-sm cursor-default leading-none">
              Tranqui<span className="text-primary">.</span>
            </h1>
            <p className="text-gray-500 text-xl md:text-2xl leading-relaxed font-medium tracking-tight">
              Tu futuro financiero, <span className="text-primary font-bold">sin agobios</span>.
            </p>
            <p className="text-gray-400 text-base md:text-lg font-normal">
              Domina tu dinero en 5 minutos.
            </p>
          </div>

          <div className="w-full max-w-sm md:max-w-md mt-10 space-y-6 animate-fade-in-up opacity-0" style={{ animationDelay: '0.4s', animationFillMode: 'forwards' }}>
            <button
              onClick={onNext}
              className="w-full md:w-auto min-w-[240px] py-4 px-8 rounded-2xl bg-primary text-white font-bold text-lg text-center shadow-lg shadow-primary/30 hover:bg-primary-hover hover:scale-[1.02] hover:shadow-primary/40 transition-all duration-300 transform active:scale-[0.98]"
            >
              Crear mi plan ahora
            </button>

            <div className="flex flex-col md:flex-row items-center gap-4 text-sm text-gray-400 font-medium pl-1">
              <div className="flex items-center gap-1.5">
                <span className="material-symbols-outlined text-sm filled text-gray-400">lock</span>
                <span>+10,000 usuarios confían en Tranqui</span>
              </div>
            </div>

            <div className="flex items-center justify-center md:justify-start pt-2">
              <button
                onClick={onLogin}
                className="text-gray-500 font-medium hover:text-gray-900 transition-colors text-sm flex items-center gap-1 p-2 group"
              >
                ¿Ya tienes cuenta? <span className="text-primary font-bold group-hover:underline decoration-2 underline-offset-4">Iniciar sesión</span>
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Visuals (Desktop Only) */}
        <div className="hidden md:flex flex-1 items-center justify-end h-full relative pointer-events-none z-10 pl-12 perspective-1000">

          <div className="flex flex-col gap-6 w-full max-w-sm">
            {/* Card 1: Security */}
            <div
              className="bg-white/95 backdrop-blur-md p-6 rounded-[2rem] shadow-xl shadow-indigo-900/5 border border-white flex items-center gap-5 transform transition-all hover:-translate-y-1.5 hover:scale-[1.02] duration-300 hover:shadow-2xl hover:shadow-indigo-500/10 opacity-0 animate-fade-in-up group/card"
              style={{ animationDelay: '0.6s', animationFillMode: 'forwards' }}
            >
              <div className="w-14 h-14 rounded-2xl bg-indigo-50 flex items-center justify-center flex-shrink-0 text-indigo-500 shadow-sm group-hover/card:scale-110 transition-transform duration-300">
                <span className="material-symbols-outlined text-3xl">shield</span>
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg tracking-tight">100% Fiable</h3>
                <p className="text-sm text-gray-500 mt-1 font-medium leading-relaxed">Recomendación personalizada basada en cálculos avanzados</p>
              </div>
            </div>

            {/* Card 2: Speed */}
            <div
              className="bg-white/95 backdrop-blur-md p-6 rounded-[2rem] shadow-xl shadow-emerald-900/5 border border-white flex items-center gap-5 transform transition-all hover:-translate-y-1.5 hover:scale-[1.02] duration-300 hover:shadow-2xl hover:shadow-emerald-500/10 opacity-0 animate-fade-in-up group/card"
              style={{ animationDelay: '0.8s', animationFillMode: 'forwards' }}
            >
              <div className="w-14 h-14 rounded-2xl bg-emerald-50 flex items-center justify-center flex-shrink-0 text-emerald-500 shadow-sm group-hover/card:scale-110 transition-transform duration-300">
                <span className="material-symbols-outlined text-3xl">bolt</span>
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg tracking-tight">Rápido y sencillo</h3>
                <p className="text-sm text-gray-500 mt-1 font-medium leading-relaxed">Tu plan financiero en 5 minutos</p>
              </div>
            </div>

            {/* Card 3: Privacy */}
            <div
              className="bg-white/95 backdrop-blur-md p-6 rounded-[2rem] shadow-xl shadow-purple-900/5 border border-white flex items-center gap-5 transform transition-all hover:-translate-y-1.5 hover:scale-[1.02] duration-300 hover:shadow-2xl hover:shadow-purple-500/10 opacity-0 animate-fade-in-up group/card"
              style={{ animationDelay: '1s', animationFillMode: 'forwards' }}
            >
              <div className="w-14 h-14 rounded-2xl bg-purple-50 flex items-center justify-center flex-shrink-0 text-purple-500 shadow-sm group-hover/card:scale-110 transition-transform duration-300">
                <span className="material-symbols-outlined text-3xl">lock</span>
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg tracking-tight">Privacidad total</h3>
                <p className="text-sm text-gray-500 mt-1 font-medium leading-relaxed">Tus datos nunca se comparten</p>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Disclaimer Footer */}
      <div className="absolute bottom-4 left-0 right-0 z-20 text-right pointer-events-none">
        <p className="text-[10px] text-white/80 font-medium px-4">
          Tranqui no ofrece asesoramiento financiero regulado. Si tienes más preguntas, no dudes en contactar a un asesor financiero.
        </p>
      </div>
    </div>
  );
};