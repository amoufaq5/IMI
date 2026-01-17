'use client'

import Link from 'next/link'
import { 
  Stethoscope, 
  Pill, 
  FileText, 
  Shield, 
  ArrowRight,
  MessageCircle,
  Building2,
  Users
} from 'lucide-react'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/80 backdrop-blur-md z-50 border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold gradient-text">UMI</span>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <Link href="#features" className="text-gray-600 hover:text-violet-600 transition">
                Features
              </Link>
              <Link href="#solutions" className="text-gray-600 hover:text-violet-600 transition">
                Solutions
              </Link>
              <Link href="#about" className="text-gray-600 hover:text-violet-600 transition">
                About
              </Link>
            </div>
            
            <div className="flex items-center space-x-4">
              <Link href="/login" className="btn-ghost">
                Sign In
              </Link>
              <Link href="/register" className="btn-primary">
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center px-4 py-2 bg-violet-50 rounded-full text-violet-600 text-sm font-medium mb-6">
              <Shield className="w-4 h-4 mr-2" />
              MHRA & UAE MOH Compliant
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              Universal Medical
              <span className="gradient-text"> Intelligence</span>
            </h1>
            
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              AI-powered healthcare platform for patients, pharmaceutical companies, 
              and healthcare providers. Get instant medical guidance and automate 
              QA/QC documentation.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/register" className="btn-primary text-lg px-8 py-3 flex items-center">
                Start Free Consultation
                <ArrowRight className="w-5 h-5 ml-2" />
              </Link>
              <Link href="/pharma" className="btn-secondary text-lg px-8 py-3">
                Pharma Solutions
              </Link>
            </div>
          </div>
          
          {/* Hero Image/Illustration */}
          <div className="mt-16 relative">
            <div className="absolute inset-0 gradient-bg opacity-5 rounded-3xl"></div>
            <div className="relative bg-white rounded-3xl shadow-soft border border-gray-100 p-8">
              <div className="grid md:grid-cols-3 gap-8">
                <div className="text-center p-6">
                  <div className="w-16 h-16 bg-violet-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <MessageCircle className="w-8 h-8 text-violet-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Consultations</h3>
                  <p className="text-gray-600">ASMETHOD protocol-based symptom assessment</p>
                </div>
                
                <div className="text-center p-6">
                  <div className="w-16 h-16 bg-violet-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <FileText className="w-8 h-8 text-violet-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">QA/QC Documents</h3>
                  <p className="text-gray-600">AI-generated SOPs, validation protocols</p>
                </div>
                
                <div className="text-center p-6">
                  <div className="w-16 h-16 bg-violet-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Pill className="w-8 h-8 text-violet-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Drug Information</h3>
                  <p className="text-gray-600">Interactions, dosages, OTC recommendations</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Powerful Features for Everyone
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              From patients seeking guidance to pharmaceutical companies needing compliance
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="card-hover">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${feature.bgColor}`}>
                  <feature.icon className={`w-6 h-6 ${feature.iconColor}`} />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solutions Section */}
      <section id="solutions" className="py-20">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Solutions for Every Stakeholder
            </h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            {/* Patients */}
            <div className="card-hover">
              <div className="flex items-start space-x-4">
                <div className="w-14 h-14 bg-violet-100 rounded-2xl flex items-center justify-center flex-shrink-0">
                  <Users className="w-7 h-7 text-violet-600" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">For Patients</h3>
                  <p className="text-gray-600 mb-4">
                    Get instant symptom assessment, drug information, and health guidance 
                    powered by medical AI.
                  </p>
                  <ul className="space-y-2 text-gray-600">
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-violet-500 rounded-full mr-3"></div>
                      ASMETHOD consultation protocol
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-violet-500 rounded-full mr-3"></div>
                      Danger sign detection
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-violet-500 rounded-full mr-3"></div>
                      OTC recommendations
                    </li>
                  </ul>
                  <Link href="/register" className="btn-primary mt-6 inline-flex items-center">
                    Start Consultation
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Link>
                </div>
              </div>
            </div>
            
            {/* Pharma */}
            <div className="card-hover">
              <div className="flex items-start space-x-4">
                <div className="w-14 h-14 bg-violet-100 rounded-2xl flex items-center justify-center flex-shrink-0">
                  <Building2 className="w-7 h-7 text-violet-600" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">For Pharmaceutical</h3>
                  <p className="text-gray-600 mb-4">
                    Automate QA/QC documentation, ensure compliance, and streamline 
                    manufacturing processes.
                  </p>
                  <ul className="space-y-2 text-gray-600">
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-violet-500 rounded-full mr-3"></div>
                      AI-generated SOPs
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-violet-500 rounded-full mr-3"></div>
                      Cleaning validation protocols
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-violet-500 rounded-full mr-3"></div>
                      Batch record management
                    </li>
                  </ul>
                  <Link href="/pharma" className="btn-primary mt-6 inline-flex items-center">
                    Explore Pharma
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4">
          <div className="gradient-bg rounded-3xl p-12 text-center text-white">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to Transform Healthcare?
            </h2>
            <p className="text-xl text-violet-100 mb-8 max-w-2xl mx-auto">
              Join thousands of users who trust UMI for medical guidance and 
              pharmaceutical compliance.
            </p>
            <Link href="/register" className="inline-flex items-center bg-white text-violet-600 font-semibold px-8 py-3 rounded-lg hover:bg-violet-50 transition">
              Get Started Free
              <ArrowRight className="w-5 h-5 ml-2" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-10 h-10 bg-violet-600 rounded-xl flex items-center justify-center">
                  <Stethoscope className="w-6 h-6 text-white" />
                </div>
                <span className="text-xl font-bold">UMI</span>
              </div>
              <p className="text-gray-400">
                Universal Medical Intelligence - AI-powered healthcare platform.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/consultation" className="hover:text-white transition">Consultations</Link></li>
                <li><Link href="/drugs" className="hover:text-white transition">Drug Info</Link></li>
                <li><Link href="/pharma" className="hover:text-white transition">Pharma QA/QC</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/about" className="hover:text-white transition">About</Link></li>
                <li><Link href="/contact" className="hover:text-white transition">Contact</Link></li>
                <li><Link href="/careers" className="hover:text-white transition">Careers</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/privacy" className="hover:text-white transition">Privacy Policy</Link></li>
                <li><Link href="/terms" className="hover:text-white transition">Terms of Service</Link></li>
                <li><Link href="/compliance" className="hover:text-white transition">Compliance</Link></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
            <p>&copy; 2024 UMI - Universal Medical Intelligence. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

const features = [
  {
    icon: MessageCircle,
    title: 'AI Consultations',
    description: 'Get instant symptom assessment using the ASMETHOD protocol with danger sign detection.',
    bgColor: 'bg-violet-100',
    iconColor: 'text-violet-600',
  },
  {
    icon: Pill,
    title: 'Drug Information',
    description: 'Search drugs, check interactions, get dosage information and OTC recommendations.',
    bgColor: 'bg-purple-100',
    iconColor: 'text-purple-600',
  },
  {
    icon: FileText,
    title: 'QA/QC Documents',
    description: 'AI-generated SOPs, cleaning validation protocols, and batch records.',
    bgColor: 'bg-violet-100',
    iconColor: 'text-violet-600',
  },
  {
    icon: Shield,
    title: 'Regulatory Compliance',
    description: 'MHRA and UAE MOH compliant documentation and processes.',
    bgColor: 'bg-purple-100',
    iconColor: 'text-purple-600',
  },
  {
    icon: Stethoscope,
    title: 'Medical Imaging',
    description: 'AI-powered analysis of X-rays, CT scans, and skin lesions.',
    bgColor: 'bg-violet-100',
    iconColor: 'text-violet-600',
  },
  {
    icon: Building2,
    title: 'Facility Management',
    description: 'Manage pharmaceutical facilities, track compliance, and production batches.',
    bgColor: 'bg-purple-100',
    iconColor: 'text-purple-600',
  },
]
