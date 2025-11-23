/* CSV ➜ image files  (global Red Hot LUT, single disk read)
*/

importPackage(Packages.ij);
importPackage(Packages.ij.process);
importPackage(Packages.ij.plugin);
importPackage(Packages.ij.gui);
importPackage(Packages.ij.measure);
importPackage(Packages.java.io);
importPackage(Packages.java.util);

// ------- choose folders ---------------------------------------------
var inDir  = IJ.getDirectory("Choose INPUT folder (CSV)");  if (inDir == null) quit();
var outDir = IJ.getDirectory("Choose OUTPUT folder");       if (outDir == null) quit();

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

gd.addChoice("Output format:",         fmts,   "Tiff");

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
var outFmt    = gd.getNextChoice();   // "Tiff", "PNG", "JPEG", "BMP"

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

// ------- helper: CSV → ImagePlus ------------------------------------
var DELIM = ",";
function loadCsv(path) {
    var rows = [], line;
    var br = new BufferedReader(new FileReader(path));
    while ((line = br.readLine()) != null) {
        line = line.trim();
        if (line != "") rows.push(line);
    }
    br.close();
    if (rows.length == 0) throw "empty";

    var w = rows[0].split(DELIM).length;
    var h = rows.length;
    var pix = java.lang.reflect.Array.newInstance(java.lang.Float.TYPE, w * h);
    var p = 0;
    for (var y = 0; y < h; y++) {
        var parts = rows[y].split(DELIM);
        if (parts.length != w) throw "width mismatch";
        for (var x = 0; x < w; x++) pix[p++] = parseFloat(parts[x]);
    }
    return new ImagePlus(new File(path).getName(), new FloatProcessor(w, h, pix, null));
}

// ------- PASS 1: open all CSVs, collect global min/max --------------
var imgs = new ArrayList();
var gMin =  java.lang.Float.POSITIVE_INFINITY;
var gMax = -java.lang.Float.POSITIVE_INFINITY;

var files = new File(inDir).listFiles();
for (var i = 0; i < files.length; i++) {
    var f = files[i];
    if (!f.isFile() || !f.getName().toLowerCase().endsWith(".csv")) continue;

    var imp;
    try {
        imp = loadCsv(f.getAbsolutePath());
    } catch (err) {
        IJ.log("Skip " + f.getName() + " (" + err + ")");
        continue;
    }

    var cal = new Calibration();
    cal.pixelWidth = cal.pixelHeight = pixSize;
    cal.setUnit("µm");
    imp.setCalibration(cal);

    var st = imp.getStatistics();
    if (st.min < gMin) gMin = st.min;
    if (st.max > gMax) gMax = st.max;

    imgs.add(imp);
}

if (imgs.isEmpty()) {
    IJ.error("No CSV files opened.");
    quit();
}
IJ.log("Global min = " + gMin + ", max = " + gMax);

// ------- PASS 2: process & save -------------------------------------
var it = imgs.iterator();
while (it.hasNext()) {
    var imp   = it.next();
    var title = imp.getTitle();
    imp.show();

    // 1) global display range + Red Hot LUT
    imp.setDisplayRange(gMin, gMax);
    IJ.run(imp, "Red Hot", "");

    // 2) optional color bar as overlay on this imp
    if (addCBar) {
        IJ.run(
            imp,
            "Calibration Bar...",
            "location=[" + cbLoc + "] fill=Black label=White number=5 decimal=1 font=" + fSize +
            (barBold ? " bold" : "") +
            " barwidth=10 barheight=128 minimum=" + gMin + " maximum=" + gMax +
            " overlay"
        );
    }

    // 3) convert to RGB (LUT + overlay stay attached)
    IJ.run(imp, "RGB Color", "");

    // 4) optional scale bar, also as overlay
    if (addScale) {
        IJ.run(imp, "Scale Bar...", sbOpts);
    }

    // 5) flatten overlays (color bar + scale bar) into pixels
    IJ.run(imp, "Flatten", "");
    imp = IJ.getImage();   // Flatten makes a new window; grab it

    // 6) save in chosen format
    var outName = title.replaceFirst("[.][cC][sS][vV]$", "") + ext;
    IJ.saveAs(imp, outFmt, outDir + outName);
    imp.close();
}

IJ.showMessage(
    "Done",
    "Saved to: " + outDir +
    "\nFormat: " + outFmt +
    "\nGlobal min = " + gMin +
    "\nGlobal max = " + gMax
);

