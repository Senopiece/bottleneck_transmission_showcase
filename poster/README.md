# Industrial Immersion poster

Final deliverables:

- `optical_receiver_poster.pptx` -- A0 portrait PowerPoint file for submission/printing.
- `optical_receiver_poster.pdf` -- compiled A0 poster for review and printing.
- `optical_receiver_poster.svg` -- vector artwork used by both outputs.
- `optical_receiver_poster.html` -- browser-printable source.

The poster follows `Poster Guidelines 2026.pdf` and the section layout from
`Poster Example 2026.pdf`. Project content is based on `report_summer.pdf`.

Rebuild from the repository root:

```powershell
cd poster
npm.cmd install
npm.cmd run build
```

The build requires Microsoft Edge for PDF export. The PowerPoint file contains
the same vector A0 artwork as the PDF, so the two deliverables remain visually
identical.
