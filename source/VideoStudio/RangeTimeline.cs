using System;
using System.Globalization;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
namespace Cutter {
    public class RangeTimeline : FrameworkElement {
        public double Duration = 1, StartTime, EndTime = 1, Position;
        public ImageSource Thumbnails;
        public List<double[]> ProtectedRanges = new List<double[]>();
        public Action<bool, double> BoundaryChanged;
        public Action<double> SeekRequested;
        int dragging; const double MarginX = 10;
        readonly Brush accent = new SolidColorBrush(Color.FromRgb(194, 243, 107));
        public RangeTimeline() { Focusable = true; Cursor = Cursors.Hand; }
        double WidthUsable { get { return Math.Max(1, ActualWidth - MarginX * 2); } }
        double X(double time) { return MarginX + Math.Max(0, Math.Min(1, time / Duration)) * WidthUsable; }
        double TimeAt(double x) { return Math.Max(0, Math.Min(1, (x - MarginX) / WidthUsable)) * Duration; }
        protected override void OnRender(DrawingContext dc) {
            base.OnRender(dc); double height = Math.Max(30, ActualHeight - 22);
            var track = new Rect(MarginX, 3, WidthUsable, height);
            dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(41, 56, 34)), null, track);
            if (Thumbnails != null) dc.DrawImage(Thumbnails, track);
            foreach(double[] range in ProtectedRanges){double left=X(range[0]),right=X(range[1]);dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(185,240,169,76)),null,new Rect(left,3,Math.Max(0,right-left),height));}
            double a = X(StartTime), b = X(EndTime);
            var shade = new SolidColorBrush(Color.FromArgb(175, 5, 10, 5));
            dc.DrawRectangle(shade, null, new Rect(MarginX, 3, Math.Max(0, a - MarginX), height));
            dc.DrawRectangle(shade, null, new Rect(b, 3, Math.Max(0, ActualWidth - MarginX - b), height));
            dc.DrawRectangle(null, new Pen(accent, 2), new Rect(a, 3, Math.Max(0, b - a), height));
            foreach (double edge in new [] { a, b }) { dc.DrawRoundedRectangle(accent, null, new Rect(edge - 6, 0, 12, height + 6), 3, 3); dc.DrawLine(new Pen(Brushes.DarkGreen, 1), new Point(edge - 1.5, height / 2 - 5), new Point(edge - 1.5, height / 2 + 5)); dc.DrawLine(new Pen(Brushes.DarkGreen, 1), new Point(edge + 1.5, height / 2 - 5), new Point(edge + 1.5, height / 2 + 5)); }
            dc.DrawLine(new Pen(Brushes.White, 1.5), new Point(X(Position), 0), new Point(X(Position), height + 7));
            for (int i = 0; i <= 4; i++) { var t = TimeSpan.FromSeconds(Duration * i / 4); string label = t.TotalHours >= 1 ? t.ToString(@"hh\:mm\:ss") : t.ToString(@"mm\:ss"); var text = new FormattedText(label, CultureInfo.CurrentCulture, FlowDirection.LeftToRight, new Typeface("Segoe UI"), 9, new SolidColorBrush(Color.FromRgb(150, 163, 142))); double x = MarginX + WidthUsable * i / 4; if (i == 4) x -= text.Width; dc.DrawText(text, new Point(x, height + 10)); }
        }
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e) {
            base.OnMouseLeftButtonDown(e); if (!IsEnabled) return; Focus(); double x = e.GetPosition(this).X;
            dragging = Math.Abs(x - X(StartTime)) < 12 ? 1 : Math.Abs(x - X(EndTime)) < 12 ? 2 : 0;
            if (dragging != 0) { CaptureMouse(); if (BoundaryChanged != null) BoundaryChanged(dragging == 1, TimeAt(x)); }
            else if (SeekRequested != null) SeekRequested(TimeAt(x)); e.Handled = true;
        }
        protected override void OnMouseMove(MouseEventArgs e) { base.OnMouseMove(e); if (IsMouseCaptured && dragging != 0 && BoundaryChanged != null) BoundaryChanged(dragging == 1, TimeAt(e.GetPosition(this).X)); }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e) { base.OnMouseLeftButtonUp(e); ReleaseMouseCapture(); }
        protected override void OnKeyDown(KeyEventArgs e) { if ((e.Key == Key.Left || e.Key == Key.Right) && dragging != 0 && BoundaryChanged != null) { double step = Keyboard.Modifiers.HasFlag(ModifierKeys.Shift) ? 1 : .1; BoundaryChanged(dragging == 1, (dragging == 1 ? StartTime : EndTime) + (e.Key == Key.Left ? -step : step)); e.Handled = true; } base.OnKeyDown(e); }
    }

}
