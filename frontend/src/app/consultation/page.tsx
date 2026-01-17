'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  Stethoscope, 
  Send, 
  ArrowLeft,
  AlertTriangle,
  CheckCircle,
  Loader2,
  User,
  Bot
} from 'lucide-react'
import { consultationApi } from '@/lib/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  dangerSigns?: string[]
}

export default function ConsultationPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm your UMI health assistant. I'll help assess your symptoms using the ASMETHOD protocol. This ensures I gather all the important information to provide you with the best guidance.\n\nPlease describe what symptoms or health concerns you're experiencing today.",
      timestamp: new Date(),
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [consultationId, setConsultationId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      let response
      
      if (!consultationId) {
        // Start new consultation
        response = await consultationApi.start(input)
        setConsultationId(response.id)
      } else {
        // Continue consultation
        response = await consultationApi.sendMessage(consultationId, input)
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.ai_response || response.message || "I understand. Let me help you with that.",
        timestamp: new Date(),
        dangerSigns: response.danger_signs,
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      // Mock response for demo
      const mockResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: getMockResponse(input),
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, mockResponse])
    } finally {
      setIsLoading(false)
    }
  }

  const getMockResponse = (userInput: string): string => {
    const input = userInput.toLowerCase()
    
    if (input.includes('headache') || input.includes('head')) {
      return "I understand you're experiencing headaches. Let me ask a few more questions:\n\n1. How long have you had this headache?\n2. Where exactly is the pain located (front, back, sides)?\n3. How would you rate the pain on a scale of 1-10?\n4. Are you experiencing any other symptoms like nausea, sensitivity to light, or vision changes?"
    }
    
    if (input.includes('fever') || input.includes('temperature')) {
      return "A fever can indicate your body is fighting an infection. To help you better:\n\n1. What is your current temperature?\n2. How long have you had the fever?\n3. Are you experiencing any other symptoms like chills, body aches, or cough?\n4. Have you taken any medication for it?"
    }
    
    if (input.includes('cough')) {
      return "Thank you for sharing that. Regarding your cough:\n\n1. Is it a dry cough or are you producing mucus?\n2. How long have you had this cough?\n3. Is it worse at any particular time of day?\n4. Do you have any other symptoms like fever, shortness of breath, or chest pain?"
    }
    
    return "Thank you for that information. To better understand your situation:\n\n1. How long have you been experiencing these symptoms?\n2. Have you taken any medications or treatments?\n3. Do you have any existing medical conditions?\n4. Are there any other symptoms you've noticed?"
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <Link href="/dashboard" className="p-2 hover:bg-gray-100 rounded-lg transition">
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </Link>
              <div className="flex items-center space-x-2">
                <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center">
                  <Stethoscope className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="font-semibold text-gray-900">Health Consultation</h1>
                  <p className="text-xs text-gray-500">ASMETHOD Protocol</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-500 hidden sm:block">Powered by UMI AI</span>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
            >
              <div className={`flex items-start space-x-3 max-w-[85%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.role === 'user' ? 'bg-violet-600' : 'bg-violet-100'
                }`}>
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-violet-600" />
                  )}
                </div>
                
                <div className={`rounded-2xl px-4 py-3 ${
                  message.role === 'user' 
                    ? 'bg-violet-600 text-white' 
                    : 'bg-white border border-gray-200 text-gray-800'
                }`}>
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  
                  {message.dangerSigns && message.dangerSigns.length > 0 && (
                    <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center text-red-700 font-medium mb-2">
                        <AlertTriangle className="w-4 h-4 mr-2" />
                        Warning Signs Detected
                      </div>
                      <ul className="text-sm text-red-600 space-y-1">
                        {message.dangerSigns.map((sign, i) => (
                          <li key={i}>• {sign}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <p className={`text-xs mt-2 ${message.role === 'user' ? 'text-violet-200' : 'text-gray-400'}`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start animate-fade-in">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-violet-600" />
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                  <div className="flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 text-violet-600 animate-spin" />
                    <span className="text-gray-500">Analyzing your symptoms...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-4">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Describe your symptoms..."
                className="w-full px-4 py-3 pr-12 rounded-xl border border-gray-200 focus:border-violet-500 focus:ring-2 focus:ring-violet-200 outline-none resize-none transition"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="btn-primary p-3 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          
          <p className="text-xs text-gray-400 mt-2 text-center">
            This is not a substitute for professional medical advice. If you're experiencing an emergency, call emergency services.
          </p>
        </div>
      </div>
    </div>
  )
}
