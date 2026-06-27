
import { motion } from 'framer-motion';
import { CheckCircle2, ShieldAlert, Cpu, Network, Zap } from 'lucide-react';

export const Models = () => {
  return (
    <section id="models" className="py-20 bg-slate-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 sm:text-4xl">Model Architectures</h2>
          <p className="mt-4 text-lg text-slate-600 max-w-2xl mx-auto">
            We implemented two updated approaches for medical VQA inference: a pretrained ConvNeXt-Tiny baseline and a hybrid BiomedCLIP pipeline fused with RotatE knowledge embeddings and GAT reasoning.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-10">
          {/* Traditional Model Card */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden card-hover"
          >
            <div className="p-8 border-b border-slate-100 bg-slate-50/50">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-slate-200 rounded-lg">
                  <Cpu className="w-6 h-6 text-slate-700" />
                </div>
                <h3 className="text-2xl font-bold text-slate-900">ConvNeXt-Tiny Baseline</h3>
              </div>
              <p className="text-slate-600">
                A pretrained visual backbone that predicts organ and modality directly from the image with a modern convolutional architecture.
              </p>
            </div>
            
            <div className="p-8 space-y-6">
              <div>
                <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Network className="w-4 h-4 text-slate-400" /> Components
                </h4>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0" />
                    <span>ConvNeXt-Tiny pretrained image encoder</span>
                  </li>
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0" />
                    <span>224x224 medical image preprocessing</span>
                  </li>
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0" />
                    <span>Dedicated organ classification head</span>
                  </li>
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0" />
                    <span>Dedicated modality classification head</span>
                  </li>
                </ul>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="block text-xs font-medium text-slate-500 mb-1">INPUTS</span>
                    <span className="text-sm text-slate-700 font-mono">Medical Image Only</span>
                  </div>
                  <div>
                    <span className="block text-xs font-medium text-slate-500 mb-1">OUTPUT</span>
                    <span className="text-sm font-semibold text-slate-700 flex items-center gap-1">
                      <ShieldAlert className="w-4 h-4" /> Organ + Modality Prediction
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Hybrid Model Card */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-white rounded-2xl shadow-lg border-2 border-primary-100 overflow-hidden card-hover relative"
          >
            <div className="absolute top-0 right-0 bg-primary-600 text-white text-xs font-bold px-3 py-1 rounded-bl-lg">
              PROPOSED
            </div>
            <div className="p-8 border-b border-primary-50 bg-primary-50/30">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-primary-100 rounded-lg">
                  <Zap className="w-6 h-6 text-primary-600" />
                </div>
                <h3 className="text-2xl font-bold text-slate-900">BiomedCLIP + RotatE + GAT</h3>
              </div>
              <p className="text-slate-600">
                A hybrid medical VQA pipeline that fuses biomedical vision-language features with retrieved knowledge-graph context for answer prediction.
              </p>
            </div>
            
            <div className="p-8 space-y-6">
              <div>
                <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Network className="w-4 h-4 text-primary-400" /> Architecture
                </h4>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-primary-500 flex-shrink-0" />
                    <span><strong>Vision-Language Encoder:</strong> BiomedCLIP for medical image and question representations</span>
                  </li>
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-primary-500 flex-shrink-0" />
                    <span><strong>Question Routing:</strong> canonical medical question templates reduce prompt mismatch</span>
                  </li>
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-primary-500 flex-shrink-0" />
                    <span><strong>Knowledge Module:</strong> RotatE entity-relation embeddings over a medical knowledge graph</span>
                  </li>
                  <li className="flex items-start gap-2 text-slate-600">
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-primary-500 flex-shrink-0" />
                    <span><strong>Graph Fusion:</strong> question-guided GAT pooling over retrieved subgraphs</span>
                  </li>
                </ul>
              </div>

              <div className="bg-primary-50/50 p-4 rounded-xl border border-primary-100">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="block text-xs font-medium text-primary-600 mb-1">INPUTS</span>
                    <span className="text-sm text-slate-700 font-mono">Medical Image + Canonical Question + Retrieved KG Subgraph</span>
                  </div>
                  <div>
                    <span className="block text-xs font-medium text-primary-600 mb-1">OUTPUT</span>
                    <div className="space-y-1">
                      <span className="text-sm font-semibold text-slate-700 flex items-center gap-1">
                        <CheckCircle2 className="w-4 h-4 text-green-500" /> Answer + Confidence
                      </span>
                      <span className="text-sm font-semibold text-slate-700 flex items-center gap-1">
                        <ShieldAlert className="w-4 h-4 text-amber-500" /> Hallucination Score
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-slate-100">
                <h4 className="text-sm font-semibold text-slate-900 mb-2">Training Configuration</h4>
                <div className="grid grid-cols-2 gap-2 text-sm text-slate-600 font-mono">
                  <div>Optimizer: AdamW</div>
                  <div>LR: 1e-4</div>
                  <div>Loss: CE + KG Margin Ranking</div>
                  <div>Epochs: 4</div>
                  <div>Batch: 8</div>
                  <div>Hardware: NVIDIA RTX 2050</div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};
