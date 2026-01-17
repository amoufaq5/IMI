'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Stethoscope, 
  MessageCircle, 
  Pill, 
  FileText, 
  Image as ImageIcon,
  Clock,
  ChevronRight,
  Plus,
  User,
  LogOut,
  Settings,
  Bell
} from 'lucide-react'
import { useAuth } from '@/lib/auth'

export default function DashboardPage() {
  const { user, logout } = useAuth()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const quickActions = [
    {
      title: 'New Consultation',
      description: 'Start an AI-powered symptom assessment',
      icon: MessageCircle,
      href: '/consultation',
      color: 'bg-violet-500',
    },
    {
      title: 'Drug Search',
      description: 'Search medications and check interactions',
      icon: Pill,
      href: '/drugs',
      color: 'bg-purple-500',
    },
    {
      title: 'Image Analysis',
      description: 'Upload medical images for AI analysis',
      icon: ImageIcon,
      href: '/imaging',
      color: 'bg-violet-600',
    },
    {
      title: 'Health Topics',
      description: 'Browse health information and guides',
      icon: FileText,
      href: '/health',
      color: 'bg-purple-600',
    },
  ]

  const recentConsultations = [
    { id: '1', title: 'Headache and fatigue', date: '2 hours ago', status: 'completed' },
    { id: '2', title: 'Skin rash consultation', date: 'Yesterday', status: 'completed' },
    { id: '3', title: 'Cold symptoms', date: '3 days ago', status: 'completed' },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold gradient-text">UMI</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="p-2 text-gray-500 hover:text-violet-600 hover:bg-violet-50 rounded-lg transition">
                <Bell className="w-5 h-5" />
              </button>
              
              <div className="relative">
                <button 
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 p-2 hover:bg-violet-50 rounded-lg transition"
                >
                  <div className="w-8 h-8 bg-violet-100 rounded-full flex items-center justify-center">
                    <User className="w-5 h-5 text-violet-600" />
                  </div>
                  <span className="text-sm font-medium text-gray-700 hidden sm:block">
                    {user?.email || 'User'}
                  </span>
                </button>
                
                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-100 py-2 animate-fade-in">
                    <Link href="/settings" className="flex items-center px-4 py-2 text-gray-700 hover:bg-violet-50">
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </Link>
                    <button 
                      onClick={logout}
                      className="flex items-center w-full px-4 py-2 text-gray-700 hover:bg-violet-50"
                    >
                      <LogOut className="w-4 h-4 mr-2" />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            Welcome back{user?.email ? `, ${user.email.split('@')[0]}` : ''}!
          </h1>
          <p className="text-gray-600">
            How can we help you today?
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {quickActions.map((action, index) => (
            <Link
              key={index}
              href={action.href}
              className="card-hover group"
            >
              <div className={`w-12 h-12 ${action.color} rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <action.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">{action.title}</h3>
              <p className="text-sm text-gray-600">{action.description}</p>
            </Link>
          ))}
        </div>

        {/* Two Column Layout */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Recent Consultations */}
          <div className="lg:col-span-2">
            <div className="card">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">Recent Consultations</h2>
                <Link href="/consultations" className="text-sm text-violet-600 hover:text-violet-700 flex items-center">
                  View all
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Link>
              </div>
              
              {recentConsultations.length > 0 ? (
                <div className="space-y-4">
                  {recentConsultations.map((consultation) => (
                    <Link
                      key={consultation.id}
                      href={`/consultation/${consultation.id}`}
                      className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-violet-50 transition"
                    >
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-violet-100 rounded-lg flex items-center justify-center">
                          <MessageCircle className="w-5 h-5 text-violet-600" />
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900">{consultation.title}</h3>
                          <div className="flex items-center text-sm text-gray-500">
                            <Clock className="w-3 h-3 mr-1" />
                            {consultation.date}
                          </div>
                        </div>
                      </div>
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <MessageCircle className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500 mb-4">No consultations yet</p>
                  <Link href="/consultation" className="btn-primary inline-flex items-center">
                    <Plus className="w-4 h-4 mr-2" />
                    Start Your First Consultation
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Quick Stats / Tips */}
          <div className="space-y-6">
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Health Tips</h2>
              <div className="space-y-4">
                <div className="p-4 bg-violet-50 rounded-lg">
                  <h3 className="font-medium text-violet-900 mb-1">Stay Hydrated</h3>
                  <p className="text-sm text-violet-700">Drink at least 8 glasses of water daily for optimal health.</p>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <h3 className="font-medium text-purple-900 mb-1">Regular Check-ups</h3>
                  <p className="text-sm text-purple-700">Schedule annual health screenings with your doctor.</p>
                </div>
              </div>
            </div>

            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Need Help?</h2>
              <p className="text-sm text-gray-600 mb-4">
                Our AI assistant is available 24/7 to help with your health questions.
              </p>
              <Link href="/consultation" className="btn-primary w-full flex items-center justify-center">
                <MessageCircle className="w-4 h-4 mr-2" />
                Start Consultation
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
