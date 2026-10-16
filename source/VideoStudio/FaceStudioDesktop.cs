using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using System.Windows.Forms.Integration;
using WpfControls = System.Windows.Controls;
using WpfMedia = System.Windows.Media;

[assembly: AssemblyTitle("VideoStudio")]
[assembly: AssemblyDescription("Cutter and FaceStudio in one native Windows application")]
[assembly: AssemblyProduct("VideoStudio Local")]
[assembly: AssemblyVersion("2.5.0.0")]
[assembly: AssemblyFileVersion("2.5.0.0")]

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.SetUnhandledExceptionMode(UnhandledExceptionMode.ThrowException);
        string diagnostic = args.Length == 2 && args[0] == "--diagnostic" ? args[1] : null;
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string engineConfig = Path.Combine(root, "engine-root.txt");
        if (File.Exists(engineConfig)) root = Path.GetFullPath(File.ReadAllText(engineConfig).Trim());
        if (!File.Exists(Path.Combine(root, "desktop_backend.py")) ||
            !File.Exists(Path.Combine(root, "runtime", "python", "python.exe")))
        {
            MessageBox.Show("Запускайте FaceStudio.exe из папки FaceStudio вместе с папкой runtime.",
                            "FaceStudio", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        bool created;
        string mutexName = "Local\\VideoStudioDesktop-" + root.ToLowerInvariant().GetHashCode().ToString("X");
        using (Mutex instance = new Mutex(true, mutexName, out created))
        {
            if (!created && diagnostic == null)
            {
                MessageBox.Show("FaceStudio уже запущена. Откройте её окно на панели задач.", "FaceStudio");
                return 0;
            }
            try { new System.Windows.Application { ShutdownMode = System.Windows.ShutdownMode.OnMainWindowClose }.Run(new StudioWindow(root, diagnostic)); }
            catch (Exception error)
            {
                File.WriteAllText(Path.Combine(root, "videostudio-error.log"), error.ToString(), Encoding.UTF8);
                if (diagnostic != null) File.WriteAllText(diagnostic, "{\"ok\":false,\"error\":" + new JavaScriptSerializer().Serialize(error.ToString()) + "}", new UTF8Encoding(false));
                if (diagnostic == null) MessageBox.Show(error.Message, "FaceStudio", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
        return Environment.ExitCode;
    }
}

internal sealed class RoundedButton : Button
{
    internal RoundedButton() { SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer, true); ForeColor = Color.FromArgb(239,243,233); }
    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        Rectangle rect = new Rectangle(1,1,Width-3,Height-3); int diameter = 14;
        using (System.Drawing.Drawing2D.GraphicsPath path = new System.Drawing.Drawing2D.GraphicsPath())
        {
            path.AddArc(rect.Left,rect.Top,diameter,diameter,180,90); path.AddArc(rect.Right-diameter,rect.Top,diameter,diameter,270,90); path.AddArc(rect.Right-diameter,rect.Bottom-diameter,diameter,diameter,0,90); path.AddArc(rect.Left,rect.Bottom-diameter,diameter,diameter,90,90); path.CloseFigure();
            Color fill = Enabled ? BackColor : Color.FromArgb(30,37,30);
            using (SolidBrush brush = new SolidBrush(fill)) e.Graphics.FillPath(brush,path);
            using (Pen pen = new Pen(Focused ? Color.FromArgb(194,243,107) : Color.FromArgb(70,84,60))) e.Graphics.DrawPath(pen,path);
            TextRenderer.DrawText(e.Graphics,Text,Font,rect,Enabled ? ForeColor : Color.FromArgb(117,129,109),TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.WordBreak);
        }
    }
}

