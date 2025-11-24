/* CSV ➜ image files  (global Red Hot LUT, single disk read)
*/

importPackage(Packages.ij);
importPackage(Packages.ij.process);
importPackage(Packages.ij.plugin);
importPackage(Packages.ij.gui);
importPackage(Packages.ij.measure);
importPackage(Packages.java.io);
importPackage(Packages.java.util);

// ------- choose folder (root containing subdirectories with TIFFs) ---------------------------------------------
var inDir = IJ.getDirectory("Choose INPUT folder (contains subdirs with TIFFs)"); if (inDir == null) quit();

// ------- dialog -----------------------------------------------------
var colors = ["White","Black","Red","Yellow","Green","Blue","Cyan","Magenta"];
var bgs    = ["None","Black","White"];
var locs   = ["Lower Right","Lower Left","Upper Right","Upper Left","At Selection"];
var fmts   = ["Tiff","PNG","JPEG","BMP"];

var gd = new GenericDialog("Batch settings");
gd.addNumericField("Pixel size (µm / pixel):", 0.1, 3);
gd.addNumericField("Scale-bar length (µm):",   10,  2);
gd.addNumericField("Bar height (pixels):",      5,  0);
gd.addNumericField("Font size:",               18,  0);

gd.addChoice("Scale-bar colour:",      colors, colors[0]);
gd.addChoice("Scale-bar background:",  bgs,    bgs[0]);
gd.addCheckbox("Scale-bar bold label", true);
gd.addChoice("Scale-bar location:",    locs,   locs[0]);
gd.addCheckbox("Add scale bar",        true);

gd.addChoice("Color-bar location:",    locs,   "Upper Right");
gd.addCheckbox("Add color bar",        true);
gd.addChoice("Color-bar mode:",        ["Overlay","Freestanding"], "Overlay");
gd.addChoice("Filename filter mode:",  ["None","Include","Exclude"], "None");
gd.addStringField("Filter pattern (regex):", "", 30);
 
gd.addChoice("Output format:",         fmts,   "PNG");
gd.addCheckbox("Preview images",       true);

gd.showDialog();
if (gd.wasCanceled()) quit();
 
var pixSize   = gd.getNextNumber();
var barLen    = gd.getNextNumber();
var barH      = gd.getNextNumber();
var fSize     = gd.getNextNumber();
var barCol    = gd.getNextChoice();
var barBg     = gd.getNextChoice();
var barBold   = gd.getNextBoolean();
var barLoc    = gd.getNextChoice();
var addScale  = gd.getNextBoolean();
var cbLoc     = gd.getNextChoice();
var addCBar   = gd.getNextBoolean();
var cbMode    = gd.getNextChoice();   // "Overlay" or "Freestanding"
var filterMode= gd.getNextChoice();   // "None","Include","Exclude"
var filterPat = gd.getNextString();
var outFmt    = gd.getNextChoice();   // "Tiff", "PNG", "JPEG", "BMP"
var preview   = gd.getNextBoolean();
 
// compile filter regex (case-insensitive); empty -> null
var filterRegex = (filterPat && filterPat.trim() !== "") ? new RegExp(filterPat, "i") : null;

if (!preview) IJ.runMacro("setBatchMode(true);");


// map format → extension
var ext = ".tif";
if (outFmt == "PNG")  ext = ".png";
else if (outFmt == "JPEG") ext = ".jpg";
else if (outFmt == "BMP")  ext = ".bmp";

var sbOpts = "width=" + barLen +
             " height=" + barH +
             " font=" + fSize +
             " color=" + barCol +
             " background=" + barBg +
             (barBold ? " bold" : "") +
             " location=[" + barLoc + "] unit=µm overlay";

 // ------- PASS 1: find TIFFs one level down, open them, collect global min/max --------------
var imgs = new ArrayList();
var parents = new ArrayList(); // matching parent dir for each image
var gMin =  java.lang.Float.POSITIVE_INFINITY;
var gMax = java.lang.Float.NEGATIVE_INFINITY;

IJ.log("PASS 1: Initial global range gMin=" + gMin + ", gMax=" + gMax);

var rootEntries = new File(inDir).listFiles();
if (rootEntries == null) { IJ.error("Cannot read input folder."); quit(); }

for (var i = 0; i < rootEntries.length; i++) {
    var sub = rootEntries[i];
    if (!sub.isDirectory()) continue;
    var tiffs = sub.listFiles();
    if (tiffs == null) continue;
    for (var j = 0; j < tiffs.length; j++) {
        var f = tiffs[j];
        if (!f.isFile()) continue;
        var name = f.getName().toLowerCase();
        if (!(name.endsWith(".tif") || name.endsWith(".tiff"))) continue;

        // hard requirement: only process TIFFs whose name contains "ac"
        // (e.g. "...AC....tif"); everything else is skipped regardless of regex.
        if (name.indexOf("ac") < 0) {
            IJ.log("PASS 1: Skipping (no 'AC' in name): " + f.getName());
            continue;
        }

        // filename filtering based on chosen mode
        if (filterMode == "Include" && filterRegex != null && !name.match(filterRegex)) continue;
        if (filterMode == "Exclude" && filterRegex != null && name.match(filterRegex)) continue;
        var imp = IJ.openImage(f.getAbsolutePath());
        if (imp == null) { IJ.log("Skip " + f.getName() + " (unable to open)"); continue; }
        var cal = new Calibration();
        cal.pixelWidth = cal.pixelHeight = pixSize;
        cal.setUnit("µm");
        imp.setCalibration(cal);
        var st = imp.getStatistics();
        IJ.log(
            "PASS 1: " + f.getName() +
            " stats min=" + st.min + ", max=" + st.max +
            " | globals before update gMin=" + gMin + ", gMax=" + gMax
        );
        if (st.min < gMin) gMin = st.min;
        if (st.max > gMax) gMax = st.max;
        IJ.log(
            "PASS 1: Updated globals after " + f.getName() +
            " -> gMin=" + gMin + ", gMax=" + gMax
        );
        imgs.add(imp);
        parents.add(sub.getAbsolutePath());
    }
}

if (imgs.size() == 0) {
    IJ.error("No TIFF files opened.");
    quit();
}
IJ.log("Global min = " + gMin + ", max = " + gMax);

 // If freestanding color bar requested, create one labeled bar per parent (manual draw to avoid truncation)
if (addCBar && cbMode == "Freestanding") {
    IJ.log("Creating freestanding colorbar(s) per parent directory (manual draw)");

    var parentSet = new java.util.HashSet();
    for (var pi = 0; pi < parents.size(); pi++) parentSet.add(parents.get(pi));

    var it = parentSet.iterator();
    while (it.hasNext()) {
        var p = it.next();
        var outDirFor = p + File.separator + "processed" + File.separator;
        new File(outDirFor).mkdirs();

        var cw = 32;   // ramp width
        var ch = 256;  // ramp height
        var padW = Math.max(140, Math.floor(9.0 * fSize));  // right padding for labels
        var padH = Math.max(70,  Math.floor(3.5 * fSize));  // bottom padding for bottom label
        var totalW = cw + padW;
        var totalH = ch + padH;

        // Build gradient (float) on the left; pad area initialized to gMin
        var cpix = java.lang.reflect.Array.newInstance(java.lang.Float.TYPE, totalW * totalH);
        var idx = 0;
        for (var yy = 0; yy < totalH; yy++) {
            var rel = (yy < ch) ? yy : (ch - 1);
            var val = gMax - (rel / (ch - 1)) * (gMax - gMin);
            for (var xx = 0; xx < totalW; xx++) {
                cpix[idx++] = (xx < cw && yy < ch) ? val : gMin;
            }
        }

        var fproc = new FloatProcessor(totalW, totalH, cpix, null);
        var cimp = new ImagePlus("colorbar", fproc);
        cimp.setDisplayRange(gMin, gMax);
        IJ.run(cimp, "Red Hot", "");

        // Convert to RGB and paint padding solid white
        var icb = new ImageConverter(cimp);
        icb.convertToRGB();
        var ip = cimp.getProcessor();
        ip.setColor(java.awt.Color.WHITE);
        ip.setRoi(cw, 0, padW, totalH); // right band
        ip.fill();
        ip.setRoi(0, ch, totalW, padH); // bottom band
        ip.fill();
        ip.resetRoi();

        // Draw ticks at top and bottom of the ramp
        ip.setColor(java.awt.Color.BLACK);
        ip.setLineWidth(2);
        ip.drawLine(cw - 3, 1, cw, 1);
        ip.drawLine(cw - 3, ch - 2, cw, ch - 2);

        // Labels (two only: max at top, min at bottom)
        ip.setFont(new java.awt.Font("SansSerif", java.awt.Font.PLAIN, fSize));
        var topLabel = "" + java.lang.String.format("%.2f", new java.lang.Double(gMax));
        var botLabel = "" + java.lang.String.format("%.2f", new java.lang.Double(gMin));
        ip.drawString(topLabel, cw + 8, Math.max(fSize + 8, 18));   // near top
        ip.drawString(botLabel, cw + 8, ch + Math.max(fSize, 18));  // in bottom padding

        // Save colorbar
        var cName = "colorbar" + ext;
        IJ.saveAs(cimp, outFmt, outDirFor + cName);
        cimp.changes = false;
        cimp.close();
    }
}

// ------- PASS 2: process & save -------------------------------------
for (var i = 0; i < imgs.size(); i++) {
    var imp = imgs.get(i);
    var parent = parents.get(i);
    var title = imp.getTitle();
    if (preview) imp.show();

    // 1) apply global min/max scaling then Red Hot LUT
    // directly set processor min/max so saved images use the global range
    var proc = imp.getProcessor();
    IJ.log(
        "PASS 2: Before scaling '" + title +
        "' display range: min=" + proc.getMin() + ", max=" + proc.getMax()
    );
    proc.setMinAndMax(gMin, gMax);
    IJ.log(
        "PASS 2: After  scaling '" + title +
        "' display range: min=" + proc.getMin() + ", max=" + proc.getMax()
    );
    IJ.run(imp, "Red Hot", "");
    IJ.log("PASS 2: Applied Red Hot LUT to '" + title + "'");

        // 2) optional color bar: either overlay on image or saved as a freestanding image
        if (addCBar && cbMode == "Overlay") {
            IJ.log(
                "PASS 2: Adding overlay calibration bar to '" + title +
                "' with min=" + gMin + ", max=" + gMax
            );
            IJ.run(
                imp,
                "Calibration Bar...",
                "location=[" + cbLoc + "] fill=Black label=White number=5 decimal=1 font=" + fSize +
                (barBold ? " bold" : "") +
                " barwidth=10 barheight=128 minimum=" + gMin + " maximum=" + gMax +
                " overlay"
            );
    } else if (addCBar && cbMode == "Freestanding") {
        // Freestanding colorbars were generated once per parent directory
        // before PASS 2 (to avoid duplicating unlabeled bars). Skip per-image.
        IJ.log("PASS 2: Freestanding colorbar already created for parent; skipping per-image colorbar for '" + title + "'");
    }

    // 3) convert to RGB (LUT + overlay stay attached)
    var ic = new ImageConverter(imp);
    ic.convertToRGB();

    // 4) optional scale bar, also as overlay
    if (addScale) {
        IJ.run(imp, "Scale Bar...", sbOpts);
    }

            // 5) flatten overlays (color bar + scale bar) into pixels (use API to avoid GUI dependency)
            imp = imp.flatten();
    // 6) save in 'processed' subfolder inside the parent directory
    var outDirFor = parent + File.separator + "processed" + File.separator;
    new File(outDirFor).mkdirs();
    var base = title;
    var dot = base.lastIndexOf('.');
    if (dot > 0) base = base.substring(0, dot);
    var outName = base + ext;
    IJ.saveAs(imp, outFmt, outDirFor + outName);
    imp.close();
}

if (preview) {
    IJ.showMessage(
        "Done",
        "Saved in 'processed' subfolders inside: " + inDir +
        "\nFormat: " + outFmt +
        "\nGlobal min = " + gMin +
        "\nGlobal max = " + gMax
    );
} else {
    IJ.log(
        "Done: Saved in 'processed' subfolders inside: " + inDir +
        " | Format: " + outFmt +
        " | Global min = " + gMin +
        " | Global max = " + gMax
    );
    IJ.runMacro("setBatchMode(false);");
}