import Foundation
import PDFKit
import AppKit

let args = CommandLine.arguments
guard args.count == 3, let pdf = PDFDocument(url: URL(fileURLWithPath: args[1])), pdf.pageCount == 1 else {
    fatalError("Expected a one-page local method diagram")
}
let page = pdf.page(at: 0)!
let bounds = page.bounds(for: .mediaBox)
let size = NSSize(width: 1800, height: 1800 * bounds.height / bounds.width)
let rendered = page.thumbnail(of: size, for: .mediaBox)
let bitmap = NSBitmapImageRep(data: rendered.tiffRepresentation!)!
try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: args[2]))
print("Rendered method diagram: " + args[2])
