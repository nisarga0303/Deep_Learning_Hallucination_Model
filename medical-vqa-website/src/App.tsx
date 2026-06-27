
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { Overview } from './components/Overview';
import { Models } from './components/Models';
import { Demo } from './components/Demo';
import { Results } from './components/Results';
import { Footer } from './components/Footer';

function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <main>
        <Hero />
        <Overview />
        <Models />
        <Results />
        <Demo />
      </main>
      <Footer />
    </div>
  );
}

export default App;
