import { useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, ScanSearch, CheckCircle, Image as ImageIcon, AlertTriangle } from 'lucide-react';
import { predictMedicalImage, type ModelType, type PredictionResponse } from '../lib/api';

type DemoResult = PredictionResponse | { error: string };

export const Demo = () => {
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>('hybrid');
  const [resultModel, setResultModel] = useState<ModelType | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [question, setQuestion] = useState('Which organ is shown in this medical image?');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
        setResult(null);
        setResultModel(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);
    setResult(null);

    try {
      const data = await predictMedicalImage(selectedFile, question, selectedModel);
      setResult(data);
      setResultModel(selectedModel);
    } catch (error) {
      console.error("Backend error:", error);
      setResult({
        error: error instanceof Error
          ? error.message
          : "Failed to connect to backend server. Make sure Python API is running on port 8000.",
      });
      setResultModel(null);
    } finally {
      setAnalyzing(false);
    }
  };

  const displayModel = resultModel ?? selectedModel;
  const isHybrid = displayModel === 'hybrid';

  return (
    <section id="demo" className="py-20 bg-slate-900 text-slate-50 overflow-hidden relative">
      <div className="absolute top-0 right-0 -z-10 w-[800px] h-[800px] bg-primary-900/30 rounded-full blur-3xl opacity-50 translate-x-1/3 -translate-y-1/3" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">Interactive Demo Panel</h2>
          <p className="mt-4 text-lg text-slate-400 max-w-2xl mx-auto">
            Test the models with images from your dataset. Upload a medical image, select the model architecture, and observe the prediction and hallucination risk.
          </p>
        </div>

        <div className="bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 overflow-hidden lg:flex">
          {/* Input Side */}
          <div className="lg:w-1/2 p-8 lg:border-r border-slate-700">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <Upload className="w-5 h-5 text-primary-400" /> Input Data
              </h3>
              
              <div className="flex flex-col items-end">
                <label className="text-xs text-slate-400 mb-1">Select Model</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value as 'hybrid' | 'traditional');
                    setResult(null);
                    setResultModel(null);
                  }}
                  className="bg-slate-900 border border-slate-600 rounded-lg py-1.5 px-3 text-sm text-slate-200 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
                >
                  <option value="hybrid">BiomedCLIP + RotatE + GAT</option>
                  <option value="traditional">ConvNeXt-Tiny Baseline</option>
                </select>
              </div>
            </div>
            
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">Medical Image</label>
              <div 
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed ${imagePreview ? 'border-primary-500/50' : 'border-slate-600 hover:border-primary-500'} rounded-xl p-6 text-center bg-slate-900/50 flex flex-col items-center justify-center cursor-pointer transition-colors group h-64`}
              >
                {imagePreview ? (
                  <div className="w-full h-full rounded-lg overflow-hidden relative">
                    <img src={imagePreview} alt="Uploaded medical scan" className="w-full h-full object-contain" />
                  </div>
                ) : (
                  <>
                    <Upload className="w-10 h-10 text-slate-500 group-hover:text-primary-400 mb-3 transition-colors" />
                    <p className="text-sm font-medium text-slate-300">Click to browse test images</p>
                    <p className="text-xs text-slate-500 mt-1">Supports standard image formats</p>
                  </>
                )}
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleImageUpload} 
                  accept="image/*" 
                  className="hidden" 
                />
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-300 mb-2">Question for API</label>
              <textarea
                value={question}
                onChange={(e) => {
                  setQuestion(e.target.value);
                  setResult(null);
                  setResultModel(null);
                }}
                rows={3}
                placeholder="Ask a medical VQA question..."
                className="w-full rounded-xl border border-slate-600 bg-slate-900/70 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
              />
            </div>

            <button 
              onClick={handleAnalyze}
              disabled={analyzing || !imagePreview || !question.trim()}
              className={`mt-8 w-full ${!imagePreview || !question.trim() ? 'bg-slate-700 text-slate-500 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-500 text-white shadow-lg shadow-primary-900/20'} font-bold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all`}
            >
              {analyzing ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
                  Processing via {isHybrid ? 'Hybrid Pipeline' : 'Traditional Pipeline'}...
                </>
              ) : (
                <>
                  <ScanSearch className="w-5 h-5" /> Analyze Image
                </>
              )}
            </button>
          </div>

          {/* Output Side */}
          <div className="lg:w-1/2 p-8 bg-slate-900/50">
             <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <ScanSearch className="w-5 h-5 text-cyan-400" /> Prediction Results
            </h3>

            {!result && !analyzing && (
              <div className="h-full min-h-[300px] flex flex-col items-center justify-center text-slate-500">
                <ImageIcon className="w-12 h-12 mb-4 opacity-50" />
                <p>{imagePreview ? 'Click "Analyze Image" to run inference.' : 'Upload an image from your data folder to begin.'}</p>
              </div>
            )}

            {analyzing && (
              <div className="h-full min-h-[300px] flex flex-col items-center justify-center">
                <div className="space-y-6 w-full max-w-sm">
                  <div>
                    <div className="flex justify-between text-sm text-slate-400 mb-1"><span>{isHybrid ? 'BiomedCLIP Visual Encoding' : 'ConvNeXt-Tiny Feature Extraction'}</span> <span>Done</span></div>
                    <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
                      <div className="h-full bg-primary-500 w-full"></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm text-slate-400 mb-1">
                      <span>{isHybrid ? 'RotatE + GAT Reasoning' : 'Organ / Modality Inference'}</span> 
                      <span className="animate-pulse">Running...</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden relative">
                      <motion.div 
                        className="h-full bg-cyan-500"
                        initial={{ width: "0%" }}
                        animate={{ width: "80%" }}
                        transition={{ duration: 1.5 }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {result && 'error' in result && (
              <div className="bg-red-900/50 border border-red-500 p-4 rounded-xl text-red-200">
                {result.error}
              </div>
            )}

            {result && !('error' in result) && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-4"
              >
                {/* Answer */}
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                  <span className="block text-xs font-medium text-slate-400 mb-1">ANSWER</span>
                  <span className="text-xl font-bold text-white">{result.answer || 'Not available for this model'}</span>
                </div>

                {/* Organ */}
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                  <span className="block text-xs font-medium text-slate-400 mb-1">ORGAN</span>
                  <span className="text-xl font-bold text-white">{result.organ}</span>
                </div>

                {/* Modality */}
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                  <span className="block text-xs font-medium text-slate-400 mb-1">MODALITY</span>
                  <span className="text-xl font-bold text-white">{result.modality}</span>
                </div>

                {/* Score */}
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                  <span className="block text-xs font-medium text-slate-400 mb-1">SCORE</span>
                  <span className={`text-2xl font-bold ${isHybrid ? 'text-green-400' : 'text-amber-400'}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Hallucinated */}
                <div className={`bg-slate-800/80 p-4 rounded-xl border relative overflow-hidden ${isHybrid ? 'border-green-900/50' : 'border-amber-900/50'}`}>
                  <div className={`absolute left-0 top-0 bottom-0 w-1 ${isHybrid ? 'bg-green-500' : 'bg-amber-500'}`}></div>
                  <span className="block text-xs font-medium text-slate-400 mb-2 pl-3">HALLUCINATED</span>
                  <div className="flex items-center gap-3 pl-3">
                    {isHybrid ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-amber-400" />
                    )}
                    <span className={`text-xl font-bold ${isHybrid ? 'text-green-400' : 'text-amber-400'}`}>
                      {result.risk}
                    </span>
                  </div>
                </div>

                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                  <span className="block text-xs font-medium text-slate-400 mb-1">BACKEND EVIDENCE</span>
                  <span className="text-sm text-slate-200">{result.evidence}</span>
                </div>

                <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-700">
                  <span className="block text-xs font-medium text-slate-400 mb-1">ACTIVE MODEL</span>
                  <span className="text-sm text-slate-200">
                    {isHybrid ? 'BiomedCLIP + RotatE + GAT hybrid pipeline' : 'ConvNeXt-Tiny traditional baseline'}
                  </span>
                </div>

                {result.debug && (
                  <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-700">
                    <span className="block text-xs font-medium text-slate-400 mb-1">DEBUG</span>
                    <div className="space-y-1 text-xs text-slate-300 font-mono break-all">
                      <div>file: {result.debug.filename ?? 'unknown'}</div>
                      <div>intent: {result.debug.question_intent ?? 'unknown'}</div>
                      <div>hash: {result.debug.image_sha256 ?? 'unknown'}</div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
