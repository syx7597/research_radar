"""Render the PPTX through macOS Quick Look and Chromium's PDF printer.

Keeps all source text editable in PPTX. The PDF is a locally rendered reading copy.
Uses an isolated browser profile and no network content.
"""
from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
PPTX=HERE/'雷达技术资料知识组织与可追溯问答_研究交流_5加1页.pptx'
PDF=PPTX.with_suffix('.pdf')
BUILD=HERE/'build';BUILD.mkdir(exist_ok=True)
QL=BUILD/'quicklook'
if QL.exists():shutil.rmtree(QL)
QL.mkdir()
subprocess.run(['qlmanage','-p','-o',str(QL),str(PPTX)],check=True,capture_output=True)
source=QL/(PPTX.name+'.qlpreview')
swift=BUILD/'render_attachments.swift'
swift.write_text('''import Foundation
import PDFKit
import AppKit
import CoreGraphics
let folder = URL(fileURLWithPath: CommandLine.arguments[1])
for url in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys:nil) where url.pathExtension == "pdf" {
 guard let doc=CGPDFDocument(url as CFURL),let page=doc.page(at:1) else {continue}
 let bounds=page.getBoxRect(.mediaBox)
 let context=CGContext(data:nil,width:Int(ceil(bounds.width*4)),height:Int(ceil(bounds.height*4)),bitsPerComponent:8,bytesPerRow:0,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
 context.scaleBy(x:4,y:4);context.translateBy(x:-bounds.minX,y:-bounds.minY)
 context.drawPDFPage(page)
 let bitmap=NSBitmapImageRep(cgImage:context.makeImage()!)
 let destination=folder.appendingPathComponent(url.deletingPathExtension().lastPathComponent+"_pdf.png")
 try bitmap.representation(using:.png,properties:[:])!.write(to:destination)
}
''')
subprocess.run(['swift',str(swift),str(source)],check=True)
html=(source/'Preview.html').read_text()
html=re.sub(r'(Attachment\d+)\.pdf',r'\1_pdf.png',html)
html=re.sub(r'(?<![\w-])(width|height|top|left|right|bottom|font-size)\s*:\s*(-?\d+(?:\.\d+)?)(?=\s*[;}])',lambda m:m[1]+':'+m[2]+'px',html)
html=html.replace('font-family:"Microsoft YaHei"','font-family:"Microsoft YaHei","PingFang SC",sans-serif')
css='''<style>
@page {size:10in 7.5in;margin:0;}
html,body {margin:0;padding:0;background:white;}
body {zoom:1.3333333333;}
div.slide {width:720px!important;height:540px!important;margin:0!important;box-shadow:none!important;break-after:page;}
div.slide:last-of-type {break-after:auto;}
div.loading-slide {display:none!important;}
</style>'''
html='<!doctype html>'+html.replace('</head>',css+'</head>')
render=source/'render.html';render.write_text(html)
chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
with tempfile.TemporaryDirectory(prefix='radar_deck_chrome_') as profile:
    args=[chrome,'--headless','--disable-gpu','--no-pdf-header-footer','--no-first-run','--disable-default-apps',f'--user-data-dir={profile}',f'--print-to-pdf={PDF}',render.as_uri()]
    with (BUILD/'chrome.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
print(json.dumps({'pdf':str(PDF),'renderer':'macOS Quick Look layout + isolated Chromium PDF print'},ensure_ascii=False))
