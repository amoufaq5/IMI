'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Stethoscope, 
  FileText, 
  Building2,
  ClipboardCheck,
  Package,
  AlertTriangle,
  Plus,
  Search,
  Filter,
  Download,
  ChevronRight,
  Clock,
  CheckCircle,
  XCircle,
  User,
  LogOut,
  Settings,
  Bell,
  BarChart3,
  TrendingUp
} from 'lucide-react'
import { useAuth } from '@/lib/auth'

interface Document {
  id: string
  title: string
  type: 'SOP' | 'Cleaning Validation' | 'Batch Record' | 'Compliance Report'
  status: 'draft' | 'pending' | 'approved' | 'rejected'
  facility: string
  createdAt: string
  updatedAt: string
}

interface Facility {
  id: string
  name: string
  type: string
  complianceScore: number
  lastAudit: string
}

export default function PharmaDashboardPage() {
  const { user, logout } = useAuth()
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'documents' | 'facilities' | 'compliance'>('overview')
  const [searchQuery, setSearchQuery] = useState('')

  const stats = [
    { label: 'Total Documents', value: '156', change: '+12%', icon: FileText, color: 'bg-violet-500' },
    { label: 'Active Facilities', value: '8', change: '+2', icon: Building2, color: 'bg-purple-500' },
    { label: 'Compliance Score', value: '94%', change: '+3%', icon: ClipboardCheck, color: 'bg-green-500' },
    { label: 'Pending Reviews', value: '12', change: '-5', icon: Clock, color: 'bg-amber-500' },
  ]

  const recentDocuments: Document[] = [
    { id: '1', title: 'SOP-001: Equipment Cleaning', type: 'SOP', status: 'approved', facility: 'Plant A', createdAt: '2024-01-15', updatedAt: '2024-01-18' },
    { id: '2', title: 'CV-023: Reactor Validation', type: 'Cleaning Validation', status: 'pending', facility: 'Plant B', createdAt: '2024-01-16', updatedAt: '2024-01-16' },
    { id: '3', title: 'BR-2024-001: Batch Record', type: 'Batch Record', status: 'draft', facility: 'Plant A', createdAt: '2024-01-17', updatedAt: '2024-01-17' },
    { id: '4', title: 'CR-Q1-2024: Quarterly Report', type: 'Compliance Report', status: 'rejected', facility: 'All', createdAt: '2024-01-10', updatedAt: '2024-01-14' },
  ]

  const facilities: Facility[] = [
    { id: '1', name: 'Manufacturing Plant A', type: 'API Production', complianceScore: 96, lastAudit: '2024-01-05' },
    { id: '2', name: 'Manufacturing Plant B', type: 'Formulation', complianceScore: 92, lastAudit: '2024-01-10' },
    { id: '3', name: 'Packaging Facility', type: 'Packaging', complianceScore: 98, lastAudit: '2024-01-12' },
    { id: '4', name: 'Quality Lab', type: 'Testing', complianceScore: 94, lastAudit: '2024-01-08' },
  ]

  const getStatusColor = (status: Document['status']) => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-700'
      case 'pending': return 'bg-amber-100 text-amber-700'
      case 'draft': return 'bg-gray-100 text-gray-700'
      case 'rejected': return 'bg-red-100 text-red-700'
    }
  }

  const getStatusIcon = (status: Document['status']) => {
    switch (status) {
      case 'approved': return <CheckCircle className="w-4 h-4" />
      case 'pending': return <Clock className="w-4 h-4" />
      case 'draft': return <FileText className="w-4 h-4" />
      case 'rejected': return <XCircle className="w-4 h-4" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center space-x-2">
                <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center">
                  <Stethoscope className="w-6 h-6 text-white" />
                </div>
                <span className="text-xl font-bold gradient-text">UMI</span>
              </Link>
              <span className="text-gray-300">|</span>
              <span className="text-sm font-medium text-gray-600">Pharma Dashboard</span>
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
                    {user?.email || 'Pharma Admin'}
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
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-1">Pharmaceutical QA/QC Dashboard</h1>
            <p className="text-gray-600">Manage documents, facilities, and compliance</p>
          </div>
          <div className="mt-4 sm:mt-0 flex space-x-3">
            <button className="btn-secondary flex items-center">
              <Download className="w-4 h-4 mr-2" />
              Export
            </button>
            <button className="btn-primary flex items-center">
              <Plus className="w-4 h-4 mr-2" />
              New Document
            </button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {stats.map((stat, index) => (
            <div key={index} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  <p className={`text-sm mt-1 ${stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
                    {stat.change} from last month
                  </p>
                </div>
                <div className={`w-12 h-12 ${stat.color} rounded-xl flex items-center justify-center`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="flex space-x-8">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'documents', label: 'Documents', icon: FileText },
              { id: 'facilities', label: 'Facilities', icon: Building2 },
              { id: 'compliance', label: 'Compliance', icon: ClipboardCheck },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center pb-4 px-1 border-b-2 font-medium text-sm transition ${
                  activeTab === tab.id
                    ? 'border-violet-600 text-violet-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4 mr-2" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Recent Documents */}
            <div className="lg:col-span-2">
              <div className="card">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold text-gray-900">Recent Documents</h2>
                  <Link href="/pharma/documents" className="text-sm text-violet-600 hover:text-violet-700 flex items-center">
                    View all
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </Link>
                </div>
                
                <div className="space-y-4">
                  {recentDocuments.map((doc) => (
                    <div
                      key={doc.id}
                      className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-violet-50 transition cursor-pointer"
                    >
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-violet-100 rounded-lg flex items-center justify-center">
                          <FileText className="w-5 h-5 text-violet-600" />
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900">{doc.title}</h3>
                          <div className="flex items-center text-sm text-gray-500 space-x-2">
                            <span>{doc.type}</span>
                            <span>•</span>
                            <span>{doc.facility}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getStatusColor(doc.status)}`}>
                          {getStatusIcon(doc.status)}
                          <span className="ml-1 capitalize">{doc.status}</span>
                        </span>
                        <ChevronRight className="w-5 h-5 text-gray-400" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Quick Actions & Alerts */}
            <div className="space-y-6">
              <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
                <div className="space-y-3">
                  <button className="w-full flex items-center p-3 bg-violet-50 rounded-lg hover:bg-violet-100 transition text-left">
                    <FileText className="w-5 h-5 text-violet-600 mr-3" />
                    <span className="font-medium text-violet-900">Generate SOP</span>
                  </button>
                  <button className="w-full flex items-center p-3 bg-purple-50 rounded-lg hover:bg-purple-100 transition text-left">
                    <ClipboardCheck className="w-5 h-5 text-purple-600 mr-3" />
                    <span className="font-medium text-purple-900">Cleaning Validation</span>
                  </button>
                  <button className="w-full flex items-center p-3 bg-violet-50 rounded-lg hover:bg-violet-100 transition text-left">
                    <Package className="w-5 h-5 text-violet-600 mr-3" />
                    <span className="font-medium text-violet-900">New Batch Record</span>
                  </button>
                </div>
              </div>

              <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Alerts</h2>
                <div className="space-y-3">
                  <div className="flex items-start p-3 bg-amber-50 rounded-lg">
                    <AlertTriangle className="w-5 h-5 text-amber-600 mr-3 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-amber-900">Audit Due</p>
                      <p className="text-sm text-amber-700">Plant B audit scheduled in 5 days</p>
                    </div>
                  </div>
                  <div className="flex items-start p-3 bg-red-50 rounded-lg">
                    <XCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-red-900">Document Rejected</p>
                      <p className="text-sm text-red-700">CR-Q1-2024 needs revision</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'documents' && (
          <div className="card">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900">All Documents</h2>
              <div className="mt-4 sm:mt-0 flex space-x-3">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search documents..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:border-violet-500 focus:ring-2 focus:ring-violet-200 outline-none"
                  />
                </div>
                <button className="btn-secondary flex items-center">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </button>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Document</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Type</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Facility</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Status</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {recentDocuments.map((doc) => (
                    <tr key={doc.id} className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer">
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <FileText className="w-4 h-4 text-violet-600 mr-2" />
                          <span className="font-medium text-gray-900">{doc.title}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-gray-600">{doc.type}</td>
                      <td className="py-3 px-4 text-gray-600">{doc.facility}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(doc.status)}`}>
                          {doc.status}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-gray-500">{doc.updatedAt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'facilities' && (
          <div className="grid md:grid-cols-2 gap-6">
            {facilities.map((facility) => (
              <div key={facility.id} className="card hover:shadow-lg transition cursor-pointer">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{facility.name}</h3>
                    <p className="text-sm text-gray-500">{facility.type}</p>
                  </div>
                  <div className="w-10 h-10 bg-violet-100 rounded-lg flex items-center justify-center">
                    <Building2 className="w-5 h-5 text-violet-600" />
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-gray-600">Compliance Score</span>
                      <span className="font-medium text-gray-900">{facility.complianceScore}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${facility.complianceScore >= 95 ? 'bg-green-500' : facility.complianceScore >= 90 ? 'bg-amber-500' : 'bg-red-500'}`}
                        style={{ width: `${facility.complianceScore}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Last Audit</span>
                    <span className="text-gray-900">{facility.lastAudit}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'compliance' && (
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-6">Compliance Overview</h2>
              <div className="space-y-4">
                {[
                  { label: 'GMP Compliance', score: 96, target: 95 },
                  { label: 'Documentation', score: 92, target: 90 },
                  { label: 'Equipment Validation', score: 88, target: 90 },
                  { label: 'Training Records', score: 94, target: 95 },
                ].map((item, index) => (
                  <div key={index}>
                    <div className="flex items-center justify-between text-sm mb-2">
                      <span className="text-gray-700">{item.label}</span>
                      <span className={`font-medium ${item.score >= item.target ? 'text-green-600' : 'text-amber-600'}`}>
                        {item.score}% / {item.target}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${item.score >= item.target ? 'bg-green-500' : 'bg-amber-500'}`}
                        style={{ width: `${item.score}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-6">Upcoming Audits</h2>
              <div className="space-y-4">
                {[
                  { facility: 'Plant B', type: 'Internal Audit', date: '2024-01-25', status: 'scheduled' },
                  { facility: 'Quality Lab', type: 'FDA Inspection', date: '2024-02-10', status: 'pending' },
                  { facility: 'Plant A', type: 'ISO Certification', date: '2024-02-20', status: 'scheduled' },
                ].map((audit, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <h3 className="font-medium text-gray-900">{audit.facility}</h3>
                      <p className="text-sm text-gray-500">{audit.type}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900">{audit.date}</p>
                      <p className="text-sm text-violet-600 capitalize">{audit.status}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
