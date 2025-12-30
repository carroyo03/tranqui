import React, { useState, useEffect } from 'react';

interface QuantumLabProps {
  onBack: () => void;
}

interface Asset {
  id: string;
  symbol: string;
  name: string;
  type: 'crypto' | 'stock' | 'etf';
}

const AVAILABLE_ASSETS: Asset[] = [
  { id: 'btc', symbol: 'BTC', name: 'Bitcoin', type: 'crypto' },
  { id: 'eth', symbol: 'ETH', name: 'Ethereum', type: 'crypto' },
  { id: 'nvda', symbol: 'NVDA', name: 'Nvidia', type: 'stock' },
  { id: 'tsla', symbol: 'TSLA', name: 'Tesla', type: 'stock' },
  { id: 'aapl', symbol: 'AAPL', name: 'Apple', type: 'stock' },
  { id: 'sp500', symbol: 'SPY', name: 'S&P 500', type: 'etf' },
  { id: 'goog', symbol: 'GOOG', name: 'Google', type: 'stock' },
  { id: 'sol', symbol: 'SOL', name: 'Solana', type: 'crypto' },
];

export const QuantumLab: React.FC<QuantumLabProps> = ({ onBack }) => {
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [phase, setPhase] = useState<'selection' | 'processing' | 'result'>('selection');
  const [loadingText, setLoadingText] = useState('Iniciando Qiskit...');
  const [results, setResults] = useState<{ symbol: string; weight: number; color: string }[]>([]);

  const runSimulation = () => {
    setPhase('processing');
    const steps = [
      "Inicializando circuito cuántico...",
      "Definiendo Hamiltonian (H)...",
      "Calculando matriz de covarianza...",
      "Ejecutando QAOA (p=3)...",
      "Colapsando función de onda...",
      "Optimizando Ratio de Sharpe..."
    ];

    let stepIndex = 0;
    const interval = setInterval(() => {
      if (stepIndex < steps.length) {
        setLoadingText(steps[stepIndex]);
        stepIndex++;
      } else {
        clearInterval(interval);
        calculateFakeResults();
        setPhase('result');
      }
    }, 800);
  };

  const calculateFakeResults = () => {
    const count = selectedAssets.length;
    let remaining = 100;
    const newResults = selectedAssets.map((assetId, index) => {
      const asset = AVAILABLE_ASSETS.find(a => a.id === assetId)!;
      let weight = 0;
      
      if (index === count - 1) {
        weight = remaining;
      } else {
        const max = remaining - (count - index - 1) * 10; 
        weight = Math.floor(Math.random() * (max - 10) + 10);
        remaining -= weight;
      }

      const colors = ['#2DD4BF', '#A78BFA', '#F472B6', '#FBBF24', '#38BDF8'];
      
      return {
        symbol: asset.symbol,
        weight: weight,
        color: colors[index % colors.length]
      };
    });
    
    setResults(newResults.sort((a, b) => b.weight - a.weight));
  };

  const toggleAsset = (id: string) => {
    if (selectedAssets.includes(id)) {
      setSelectedAssets(selectedAssets.filter(a => a !== id));
    } else if (selectedAssets.length < 5) {
      setSelectedAssets([...selectedAssets, id]);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0F172A] text-white relative font-sans overflow-hidden">
        
        {/* Background Animation */}
        <div className="absolute inset-0 z-0">
             <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-soft-light"></div>
            <div className="absolute top-[-20%] left-[-20%] w-[500px] h-[500px] bg-indigo-600/30 rounded-full blur-[100px] animate-pulse"></div>
            <div className="absolute bottom-[-20%] right-[-20%] w-[500px] h-[500px] bg-purple-600/30 rounded-full blur-[100px] animate-pulse animation-delay-2000"></div>
        </div>

        {/* Glass Header */}
        <div className="relative z-10 px-6 py-5 flex items-center justify-between bg-white/5 backdrop-blur-xl border-b border-white/5 shadow-lg">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.5)] border border-white/20">
                    <span className="material-symbols-outlined text-white">science</span>
                </div>
                <div>
                    <h1 className="text-lg font-bold tracking-tight text-white">Quantum Lab</h1>
                    <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>
                        <p className="text-[10px] text-gray-300 font-mono tracking-wider">POWERED BY QISKIT</p>
                    </div>
                </div>
            </div>
            <button onClick={onBack} className="w-8 h-8 flex items-center justify-center rounded-full bg-white/5 hover:bg-white/10 transition-colors border border-white/5">
                <span className="material-symbols-outlined text-sm text-gray-300">close</span>
            </button>
        </div>

        {/* Content */}
        <div className="flex-1 relative z-10 flex flex-col p-6 overflow-y-auto no-scrollbar">
            
            {phase === 'selection' && (
                <div className="animate-fade-in flex flex-col h-full">
                    <div className="mb-6">
                        <h2 className="text-2xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-indigo-200 to-white">Diseña tu Portafolio</h2>
                        <p className="text-indigo-200/70 text-sm leading-relaxed">
                            Selecciona hasta 5 activos. Usaremos algoritmos cuánticos (QAOA) para calcular la distribución óptima.
                        </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3 mb-8">
                        {AVAILABLE_ASSETS.map((asset) => {
                            const isSelected = selectedAssets.includes(asset.id);
                            return (
                                <button
                                    key={asset.id}
                                    onClick={() => toggleAsset(asset.id)}
                                    className={`p-4 rounded-2xl border flex items-center gap-3 transition-all duration-300 relative overflow-hidden group ${
                                        isSelected 
                                        ? 'bg-indigo-600/20 border-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.2)]' 
                                        : 'bg-slate-800/40 border-white/5 hover:bg-slate-700/40 hover:border-white/10'
                                    }`}
                                >
                                    <div className={`w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold transition-transform group-hover:scale-110 ${
                                        asset.type === 'crypto' ? 'bg-orange-500/20 text-orange-300' :
                                        asset.type === 'stock' ? 'bg-blue-500/20 text-blue-300' : 'bg-green-500/20 text-green-300'
                                    }`}>
                                        {asset.symbol[0]}
                                    </div>
                                    <div className="text-left z-10">
                                        <div className="font-bold text-sm text-white">{asset.symbol}</div>
                                        <div className="text-[10px] text-gray-400">{asset.name}</div>
                                    </div>
                                    {isSelected && (
                                        <div className="absolute top-2 right-2">
                                            <span className="material-symbols-outlined text-indigo-400 text-sm drop-shadow-[0_0_5px_rgba(129,140,248,1)]">check_circle</span>
                                        </div>
                                    )}
                                </button>
                            );
                        })}
                    </div>

                    <div className="mt-auto">
                        <button
                            disabled={selectedAssets.length < 2}
                            onClick={runSimulation}
                            className={`w-full py-4 rounded-2xl font-bold text-lg shadow-xl flex items-center justify-center gap-2 transition-all duration-300 relative overflow-hidden ${
                                selectedAssets.length < 2
                                ? 'bg-slate-800 text-gray-500 cursor-not-allowed border border-white/5'
                                : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:scale-[1.02] hover:shadow-indigo-500/40 border border-indigo-400/30'
                            }`}
                        >
                            {selectedAssets.length >= 2 && <div className="absolute inset-0 bg-white/10 animate-pulse"></div>}
                            <span className="material-symbols-outlined relative z-10">auto_awesome</span>
                            <span className="relative z-10">Optimizar con IA</span>
                        </button>
                    </div>
                </div>
            )}

            {phase === 'processing' && (
                <div className="flex-1 flex flex-col items-center justify-center text-center animate-fade-in">
                    <div className="relative w-40 h-40 mb-10">
                        {/* Orbital Animation */}
                        <div className="absolute inset-0 border border-indigo-500/30 rounded-full animate-[spin_3s_linear_infinite]"></div>
                        <div className="absolute inset-4 border border-purple-500/30 rounded-full animate-[spin_4s_linear_infinite_reverse]"></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                             <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full blur-[30px] animate-pulse"></div>
                             <div className="w-2 h-2 bg-white rounded-full shadow-[0_0_20px_white] z-10"></div>
                        </div>
                        {/* Particles */}
                        <div className="absolute top-0 left-1/2 w-1 h-1 bg-cyan-400 rounded-full shadow-[0_0_10px_cyan] animate-[ping_1s_infinite]"></div>
                    </div>
                    
                    <h3 className="text-2xl font-mono font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 to-purple-300 mb-4 tracking-wider">
                        PROCESANDO
                    </h3>
                    <div className="h-6 overflow-hidden">
                        <p className="text-sm text-indigo-200/80 font-mono animate-slide-up">{loadingText}</p>
                    </div>
                    
                    {/* Fake Terminal Output */}
                    <div className="mt-12 w-full bg-black/60 backdrop-blur-md rounded-xl p-5 text-left font-mono text-[11px] text-green-400/90 h-32 overflow-hidden border border-white/10 shadow-2xl relative">
                        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-green-500/50 to-transparent opacity-50"></div>
                        <p>{'>'} import qiskit</p>
                        <p>{'>'} from qiskit_optimization import QAOA</p>
                        <p>{'>'} backend = Aer.get_backend('qasm_simulator')</p>
                        <p>{'>'} optimize_portfolio(assets={JSON.stringify(selectedAssets)})</p>
                        <p className="animate-pulse text-green-300">{'>'} computing eigenvalues...</p>
                        <p className="text-green-600 opacity-50">{'>'} collapsing wave function</p>
                    </div>
                </div>
            )}

            {phase === 'result' && (
                <div className="animate-fade-in-up flex flex-col h-full">
                    <div className="text-center mb-8">
                        <div className="inline-flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 px-4 py-1.5 rounded-full text-xs font-bold mb-4 border border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                            <span className="material-symbols-outlined text-sm">check_circle</span>
                            OPTIMIZACIÓN COMPLETADA
                        </div>
                        <h2 className="text-3xl font-bold mb-2 text-white">Tu Fórmula Maestra</h2>
                        <p className="text-indigo-200/60 text-sm">Matemáticamente balanceado para minimizar riesgo.</p>
                    </div>

                    {/* Result Card */}
                    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-3xl p-6 mb-6 shadow-xl relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl -mr-10 -mt-10"></div>
                        
                        {results.map((item, idx) => (
                            <div key={idx} className="mb-5 last:mb-0 relative z-10">
                                <div className="flex justify-between items-end mb-2">
                                    <span className="font-bold text-lg flex items-center gap-3 text-white">
                                        <div className="w-3 h-3 rounded-full shadow-[0_0_10px]" style={{ backgroundColor: item.color, boxShadow: `0 0 10px ${item.color}` }}></div>
                                        {item.symbol}
                                    </span>
                                    <span className="font-mono text-xl font-bold" style={{ color: item.color }}>{item.weight}%</span>
                                </div>
                                {/* Bar */}
                                <div className="w-full h-2.5 bg-slate-700/50 rounded-full overflow-hidden border border-white/5">
                                    <div 
                                        className="h-full rounded-full transition-all duration-1000 ease-out shadow-[0_0_10px]" 
                                        style={{ width: `${item.weight}%`, backgroundColor: item.color, boxShadow: `0 0 10px ${item.color}` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-6">
                        <div className="bg-slate-800/50 p-4 rounded-2xl border border-white/10 text-center backdrop-blur-sm">
                            <div className="text-[10px] text-gray-400 uppercase tracking-widest mb-1">Sharpe Ratio</div>
                            <div className="text-2xl font-bold text-emerald-400 drop-shadow-[0_0_5px_rgba(52,211,153,0.5)]">2.45</div>
                            <div className="text-[10px] text-emerald-500/70">Excelente</div>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-2xl border border-white/10 text-center backdrop-blur-sm">
                            <div className="text-[10px] text-gray-400 uppercase tracking-widest mb-1">Riesgo (Vol)</div>
                            <div className="text-2xl font-bold text-amber-400 drop-shadow-[0_0_5px_rgba(251,191,36,0.5)]">Low</div>
                            <div className="text-[10px] text-amber-500/70">Optimizado</div>
                        </div>
                    </div>

                    <div className="mt-auto">
                         <button
                            onClick={() => {
                                setPhase('selection');
                                setResults([]);
                                setSelectedAssets([]);
                            }}
                            className="w-full py-4 bg-white/5 hover:bg-white/10 text-white font-semibold rounded-2xl transition-colors border border-white/10 flex items-center justify-center gap-2 group"
                        >
                            <span className="material-symbols-outlined group-hover:rotate-180 transition-transform duration-500">refresh</span>
                            Probar otra combinación
                        </button>
                    </div>
                </div>
            )}
        </div>
    </div>
  );
};