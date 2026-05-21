import fitz

doc_a = fitz.open("signal_without_stab.pdf")
doc_b = fitz.open("signals_with_stab.pdf")

page_a = doc_a[0]
page_b = doc_b[0]

w_a, h_a = page_a.rect.width, page_a.rect.height
w_b, h_b = page_b.rect.width, page_b.rect.height

new_w = max(w_a, w_b)
new_h = h_a + h_b

new_doc = fitz.open()
new_page = new_doc.new_page(width=new_w, height=new_h)

new_page.show_pdf_page(fitz.Rect(0, 0, w_a, h_a), doc_a, 0)
new_page.show_pdf_page(fitz.Rect(0, h_a, w_b, h_a + h_b), doc_b, 0)

fontsize = 30

ax, ay = 35, 0
new_page.insert_htmlbox(
    fitz.Rect(ax, ay, ax + 100, ay + fontsize + 10),
    f"<span style='font-size:{fontsize}px;'>(а)</span>",
)

bx, by = 35, -18
new_page.insert_htmlbox(
    fitz.Rect(bx, h_a + by, bx + 100, h_a + by + fontsize + 10),
    f"<span style='font-size:{fontsize}px;'>(б)</span>",
)

new_doc.save("signals_without_and_with_stab.pdf")