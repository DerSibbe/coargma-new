import './globals.css'
import 'bootstrap/dist/css/bootstrap.css';
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Comparative Argumentative Machine v2',
  description: 'Comparative Argumentative Machine v2',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
