using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Threading;
using Forms = System.Windows.Forms;

internal sealed partial class StudioWindow : Window
{
    private readonly string root, diagnosticPath;
    private readonly Window template;
    private readonly MediaElement media;
    private readonly Button swap, repair, cancel, last, open, emptyOpen, play, save, folder, check, sourceButton, resultButton;
    private readonly TextBox command, report;
    private readonly TextBlock stage, localStatus, fileName, fileDetails, position;
    private readonly Border dropArea, emptyPanel, dropHighlight, previewOverlay;
    private readonly Slider seek;
    private readonly ProgressBar progress;
    private readonly DispatcherTimer timer = new DispatcherTimer();
    private readonly Stopwatch alive = Stopwatch.StartNew();
    private readonly System.Media.SoundPlayer readySound;
    private Process backend, previewProcess;
    private string videoPath, projectPath, resultPath, analysisPath, analysisVideo, previewOriginal, lastError;
    private bool hasPendingRender;
    private object[] characters = new object[0], assignments = new object[0], personGroups = new object[0];
    private bool busy, loading, closed, opened, paused = true, settingSeek, cancelRequested, converted;
    private string activeAction, viewKind = "Исходник";
    private int mediaVersion, loadVersion, signalCount;
    private double videoDuration, videoFps = 25;
    private bool diagnosticRunning;
    private string diagnosticSelection;
    private readonly Dictionary<string, object> diagnostics = new Dictionary<string, object>();
    private static readonly string[] formats = { ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg", ".mxf", ".webm", ".wmv" };
    [DllImport("dwmapi.dll")] private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attribute, ref int value, int size);
    private T Find<T>(string name) where T : class { return template.FindName(name) as T; }
    private static JavaScriptSerializer Serializer() { return new JavaScriptSerializer { MaxJsonLength = Int32.MaxValue }; }

    internal StudioWindow(string root, string diagnostic)
    {
        this.root = root; diagnosticPath = diagnostic;
        using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("FaceStudio.Window.xaml")) template = (Window)XamlReader.Load(stream);
        Title = template.Title; Width = template.Width; Height = template.Height; MinWidth = template.MinWidth; MinHeight = template.MinHeight;
        Background = template.Background; Foreground = template.Foreground; FontFamily = template.FontFamily; FontSize = template.FontSize;
        Resources = template.Resources; WindowStartupLocation = WindowStartupLocation.CenterScreen;
        object content = template.Content; template.Content = null; Content = content;
        media = Find<MediaElement>("Player"); command = Find<TextBox>("Command"); report = Find<TextBox>("Report");
        swap = Find<Button>("SwapButton"); repair = Find<Button>("RepairButton"); cancel = Find<Button>("CancelButton"); last = Find<Button>("LastButton");
        open = Find<Button>("OpenButton"); emptyOpen = Find<Button>("EmptyOpenButton"); play = Find<Button>("PlayButton"); save = Find<Button>("SaveButton");
        folder = Find<Button>("FolderButton"); check = Find<Button>("CheckButton"); sourceButton = Find<Button>("SourceButton"); resultButton = Find<Button>("ResultButton");
        stage = Find<TextBlock>("Stage"); localStatus = Find<TextBlock>("LocalStatus"); fileName = Find<TextBlock>("FileName"); fileDetails = Find<TextBlock>("FileDetails"); position = Find<TextBlock>("PositionLabel");
        dropArea = Find<Border>("DropArea"); emptyPanel = Find<Border>("EmptyPanel"); dropHighlight = Find<Border>("DropHighlight"); previewOverlay = Find<Border>("PreviewOverlay");
        seek = Find<Slider>("SeekSlider"); progress = Find<ProgressBar>("Progress");
        readySound = new System.Media.SoundPlayer(Assembly.GetExecutingAssembly().GetManifestResourceStream("FaceStudio.Ready.wav")); readySound.Load();
        InitializeCutter();
        AllowDrop = true;
        PreviewDragEnter += DragOverVideo; PreviewDragOver += DragOverVideo; PreviewDragLeave += delegate { HighlightDrop(false); }; PreviewDrop += DropVideo;
        open.Click += async delegate { await BrowseVideo(); }; emptyOpen.Click += async delegate { await BrowseVideo(); };
        Find<Button>("ResumeButton").Click+=delegate {if(projectPath!=null)StartTask("resume",new Dictionary<string,object>{{"project",projectPath}});};
        swap.Click += delegate { Swap(); }; repair.Click += delegate { Execute(command.Text); }; check.Click += delegate { if (projectPath != null) StartTask("check",new Dictionary<string,object> { { "project",projectPath } }); };
        cancel.Click += delegate { CancelTask(); }; last.Click += delegate { StartTask("last", new Dictionary<string, object>()); };
        save.Click += delegate { SaveVideo(); }; folder.Click += delegate { if (resultPath != null) Process.Start("explorer.exe", "/select," + Quote(resultPath)); };
        play.Click += delegate { TogglePlay(); };
        Find<Button>("MuteButton").Click += delegate { media.IsMuted = !media.IsMuted; Find<Button>("MuteButton").Content = media.IsMuted ? "Звук выкл." : "Звук вкл."; };
        sourceButton.Click += delegate { if (!loading && videoPath != null) OpenPreview(videoPath, "Исходник"); };
        resultButton.Click += delegate { if (!loading && resultPath != null) OpenPreview(resultPath, "Результат"); };
        seek.ValueChanged += delegate { if (!settingSeek && opened && media.NaturalDuration.HasTimeSpan) media.Position = TimeSpan.FromSeconds(media.NaturalDuration.TimeSpan.TotalSeconds * seek.Value / 1000); };
        command.PreviewKeyDown += delegate(object sender, KeyEventArgs e) { if (e.Key == Key.Enter && (Keyboard.Modifiers & ModifierKeys.Control) != 0) { e.Handled = true; Execute(command.Text); } };
        PreviewKeyDown += delegate(object sender, KeyEventArgs e) { if (e.Key == Key.Space && !(Keyboard.FocusedElement is TextBox)) { e.Handled = true; TogglePlay(); } };
        media.Volume = .7;
        media.MediaOpened += delegate { opened = true; media.Pause(); paused = true; play.Content = "▶"; previewOverlay.Visibility = Visibility.Collapsed; play.IsEnabled = true; seek.IsEnabled = true; RefreshCutControls(); UpdateTime(); };
        media.MediaEnded += delegate { media.Pause(); paused = true; play.Content = "▶"; media.Position = TimeSpan.Zero; };
        media.MediaFailed += async delegate { if (!closed) await ConvertPreview(mediaVersion); };
        dropArea.SizeChanged += delegate { bool compact = dropArea.ActualHeight < 390; Find<TextBlock>("UploadIcon").Visibility = Find<TextBlock>("UploadFormats").Visibility = compact ? Visibility.Collapsed : Visibility.Visible; emptyOpen.Margin = new Thickness(0,compact ? 10 : 20,0,0); };
        SourceInitialized += delegate { try { int dark = 1; DwmSetWindowAttribute(new WindowInteropHelper(this).Handle, 20, ref dark, 4); } catch { } };
        Closing += OnClosing;
        Loaded += delegate { if (diagnosticPath != null) Hide(); StartTask("startup", new Dictionary<string, object>()); };
        timer.Interval = TimeSpan.FromMilliseconds(80); timer.Tick += delegate { UpdateTime(); if (diagnosticPath != null && !diagnosticRunning && !busy && opened) { diagnosticRunning = true; RunDiagnostics(); } }; timer.Start();
    }

    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\""); int slashes = 0;
        foreach (char c in value) { if (c == '\\') { slashes++; continue; } if (c == '"') result.Append('\\', slashes * 2 + 1).Append(c); else result.Append('\\', slashes).Append(c); slashes = 0; }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }
    private string Args(string[] values) { List<string> result = new List<string>(); foreach (string value in values) result.Add(Quote(value)); return String.Join(" ", result.ToArray()); }
    private async Task<string> RunTool(string tool, string[] args, bool preview)
    {
        using (Process process = new Process())
        {
            process.StartInfo = new ProcessStartInfo(Path.Combine(root, "runtime", "bin", tool + ".exe"), Args(args)) { UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden, RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8 };
            process.Start(); if (preview) previewProcess = process;
            Task<string> output = process.StandardOutput.ReadToEndAsync(), errors = process.StandardError.ReadToEndAsync();
            await Task.Run(delegate { process.WaitForExit(); }); string text = await output, error = await errors;
            if (previewProcess == process) previewProcess = null;
            if (process.ExitCode != 0) throw new IOException("Не удалось прочитать видео. Проверьте, что файл не повреждён и содержит видеодорожку." + (diagnosticPath != null ? "\n" + error : ""));
            return text;
        }
    }
    private async Task<Dictionary<string, object>> Probe(string path)
    {
        string text = await RunTool("ffprobe", new string[] { "-v", "error", "-show_format", "-show_streams", "-of", "json", path }, false);
        Dictionary<string, object> data = Serializer().Deserialize<Dictionary<string, object>>(text), video = null;
        foreach (object item in (System.Collections.IEnumerable)data["streams"]) { Dictionary<string, object> stream = (Dictionary<string, object>)item; if (Convert.ToString(stream["codec_type"]) == "video") { video = stream; break; } }
        if (video == null) throw new IOException("В этом файле нет видеодорожки. Выберите видео.");
        Dictionary<string, object> format = (Dictionary<string, object>)data["format"];
        double duration; if (!format.ContainsKey("duration") || !Double.TryParse(Convert.ToString(format["duration"]), NumberStyles.Float, CultureInfo.InvariantCulture, out duration) || duration <= 0 || Double.IsInfinity(duration) || Double.IsNaN(duration)) throw new IOException("Не удалось определить длительность видео.");
        double fps = 25; string[] rate = Convert.ToString(video["avg_frame_rate"]).Split('/'); double a,b;
        if (rate.Length == 2 && Double.TryParse(rate[0], NumberStyles.Float, CultureInfo.InvariantCulture, out a) && Double.TryParse(rate[1], NumberStyles.Float, CultureInfo.InvariantCulture, out b) && b > 0 && a > 0) fps = a / b;
        return new Dictionary<string, object> { { "duration", duration }, { "fps", fps }, { "width", video["width"] }, { "height", video["height"] } };
    }
    private async Task BrowseVideo()
    {
        if (busy || loading) return;
        string selected = null;
        if (diagnosticPath != null && diagnosticSelection != null) selected = diagnosticSelection;
        else using (Forms.OpenFileDialog dialog = new Forms.OpenFileDialog { Title = "Открыть видео", Filter = "Видео|*.mp4;*.mov;*.mkv;*.avi;*.m4v;*.mpeg;*.mpg;*.mxf;*.webm;*.wmv|Все файлы|*.*" }) if (dialog.ShowDialog(new WindowOwner(this)) == Forms.DialogResult.OK) selected = dialog.FileName;
        if (selected != null) await OpenVideo(selected, "button");
    }
    private static bool IsVideoPath(string path) { return File.Exists(path) && Array.IndexOf(formats, Path.GetExtension(path).ToLowerInvariant()) >= 0; }
    private void HighlightDrop(bool value) { dropHighlight.Visibility = value ? Visibility.Visible : Visibility.Collapsed; dropArea.BorderBrush = new SolidColorBrush(value ? Color.FromRgb(194,243,107) : Color.FromRgb(48,56,47)); }
    private void DragOverVideo(object sender, DragEventArgs e)
    {
        string[] files = e.Data.GetDataPresent(DataFormats.FileDrop) ? e.Data.GetData(DataFormats.FileDrop) as string[] : null;
        bool valid = !busy && !loading && files != null && files.Length == 1 && IsVideoPath(files[0]);
        // Accept the drop event for unsupported files too, so releasing one
        // produces an explanation rather than silently doing nothing.
        e.Effects = !busy && !loading && files != null && files.Length > 0 ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true; HighlightDrop(valid);
    }
    private async void DropVideo(object sender, DragEventArgs e)
    {
        e.Handled = true; HighlightDrop(false);
        string[] files = e.Data.GetDataPresent(DataFormats.FileDrop) ? e.Data.GetData(DataFormats.FileDrop) as string[] : null;
        await OpenDroppedFiles(files);
    }
    private async Task<bool> OpenDroppedFiles(string[] files)
    {
        if (busy || loading) return false;
        if (files == null || files.Length != 1) { Error("Перетащите один видеофайл."); return false; }
        return await OpenVideo(files[0], "drop");
    }
    private async Task<bool> OpenVideo(string path, string method)
    {
        if (busy || loading) return false;
        if (!IsVideoPath(path)) { Error("Этот файл не подходит. Выберите видео MP4, MOV, MKV, AVI, WEBM, M4V, MPEG, MXF или WMV."); return false; }
        loading = true; RefreshControls(); stage.Text = "Открываем видео…"; int version = ++loadVersion;
        try
        {
            Dictionary<string, object> info = await Probe(path);
            if (closed || version != loadVersion) return false;
            videoPath = Path.GetFullPath(path); videoDuration = Convert.ToDouble(info["duration"]); videoFps = Convert.ToDouble(info["fps"]);
            projectPath = resultPath = analysisPath = analysisVideo = null; characters = new object[0]; assignments = new object[0]; personGroups = new object[0];
            hasPendingRender=false;LoadProtection(videoPath);
            command.Clear(); report.Text = "Видео открыто. Нажмите «Заменить лица», чтобы найти людей и назначить фотографии.";
            fileName.Text = Path.GetFileName(videoPath); fileName.ToolTip = videoPath;
            fileDetails.Text = "Исходник · " + Stamp(TimeSpan.FromSeconds(videoDuration)) + " · " + info["width"] + " × " + info["height"];
            lastError = null; stage.Text = "Видео готово. Замена запускается кнопкой «Заменить лица».";
            diagnostics["last_load_method"] = method; OpenPreview(videoPath, "Исходник");
            return true;
        }
        catch (Exception error) { if (!closed) Error(error.Message); return false; }
        finally { loading = false; if (!closed) RefreshControls(); }
    }
    private void OpenPreview(string path, string kind)
    {
        StopPreviewConversion(); ++mediaVersion; previewOriginal = path; viewKind = kind; converted = false;
        media.Close(); opened = false; paused = true; play.Content = "▶"; play.IsEnabled = seek.IsEnabled = false;
        settingSeek = true; seek.Value = 0; settingSeek = false;
        emptyPanel.Visibility = Visibility.Collapsed; previewOverlay.Visibility = Visibility.Visible;
        Find<TextBlock>("PreviewMessage").Text = "Открываем предпросмотр…";
        fileName.Text = kind + " · " + Path.GetFileName(kind == "Результат" && videoPath != null ? videoPath : path);
        fileName.ToolTip = path;
        media.Source = new Uri(path, UriKind.Absolute); media.Play();
        Task unusedCut = PrepareCut(path,mediaVersion);
    }
    private async Task ConvertPreview(int version)
    {
        if (closed || version != mediaVersion) return;
        if (converted) { previewOverlay.Visibility = Visibility.Visible; Find<TextBlock>("PreviewMessage").Text = "Предпросмотр недоступен. Готовое видео можно сохранить и открыть из папки."; return; }
        converted = true; string original = previewOriginal;
        previewOverlay.Visibility = Visibility.Visible; Find<TextBlock>("PreviewMessage").Text = "Подготавливаем совместимый предпросмотр…";
        string folder = Path.Combine(root, "data", "previews"); Directory.CreateDirectory(folder); string output = Path.Combine(folder, Guid.NewGuid().ToString("N") + ".mp4");
        try
        {
            await RunTool("ffmpeg", new string[] { "-v", "error", "-y", "-i", original, "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", output }, true);
            if (closed || version != mediaVersion) return;
            media.Close(); media.Source = new Uri(output); media.Play();
        }
        catch (Exception error) { if (!closed && version == mediaVersion) { Find<TextBlock>("PreviewMessage").Text = "Не удалось подготовить предпросмотр. " + error.Message; } }
    }
    private void StopPreviewConversion() { Process process = previewProcess; previewProcess = null; if (process != null) { try { if (!process.HasExited) process.Kill(); } catch (InvalidOperationException) { } } }
    private void TogglePlay() { if (!opened) return; if (paused) { media.Play(); paused = false; play.Content = "Ⅱ"; } else { media.Pause(); paused = true; play.Content = "▶"; } }
    private string Stamp(TimeSpan time) { return ((int)time.TotalMinutes).ToString() + ":" + time.Seconds.ToString("00"); }
    private void UpdateTime()
    {
        if (!opened || !media.NaturalDuration.HasTimeSpan) return;
        TimeSpan duration = media.NaturalDuration.TimeSpan; position.Text = Stamp(media.Position) + " / " + Stamp(duration);
        if (!seek.IsMouseCaptureWithin && duration.TotalSeconds > 0) { settingSeek = true; seek.Value = Math.Max(0, Math.Min(1000, media.Position.TotalSeconds * 1000 / duration.TotalSeconds)); settingSeek = false; }
        fileDetails.Text = viewKind + " · " + Stamp(duration) + " · " + media.NaturalVideoWidth + " × " + media.NaturalVideoHeight;
        TickCutter();
    }
    private void RefreshControls()
    {
        bool available = !busy && !loading;
        open.IsEnabled = emptyOpen.IsEnabled = last.IsEnabled = available;
        swap.IsEnabled = available && videoPath != null;
        repair.IsEnabled = command.IsEnabled = check.IsEnabled = available && resultPath != null;
        save.IsEnabled = folder.IsEnabled = resultButton.IsEnabled = resultPath != null;
        sourceButton.IsEnabled = videoPath != null && !loading;
        Find<Button>("ResumeButton").Visibility=hasPendingRender?Visibility.Visible:Visibility.Collapsed;
        Find<Button>("ResumeButton").IsEnabled=available&&hasPendingRender&&projectPath!=null;
        cancel.IsEnabled = busy && (activeAction == "swap" || activeAction == "execute" || activeAction == "scan" || activeAction == "check" || activeAction == "resume") && !cancelRequested;
        if (!busy) progress.IsIndeterminate = false;
        RefreshCutControls();
    }
    private static object[] Items(object value) { List<object> values = new List<object>(); if (value != null) foreach (object item in (System.Collections.IEnumerable)value) values.Add(item); return values.ToArray(); }
    private void Swap()
    {
        if (busy || loading || videoPath == null) return;
        if (analysisVideo == videoPath && File.Exists(analysisPath) && characters.Length > 0 && HasPeopleProfiles()) { ChooseAssignments(); return; }
        Dictionary<string,object> scanJob=new Dictionary<string,object> { { "video", videoPath }, { "identity_analysis", true } };
        if(analysisVideo==videoPath && File.Exists(analysisPath) && characters.Length>0)
            scanJob["previous"]=new Dictionary<string,object>{{"analysis",analysisPath},{"assignments",assignments},{"person_groups",personGroups}};
        StartTask("scan",scanJob);
    }
    private bool HasPeopleProfiles()
    {
        try { Dictionary<string,object> data=Serializer().Deserialize<Dictionary<string,object>>(File.ReadAllText(analysisPath)); return data.ContainsKey("identity_schema") && Convert.ToInt32(data["identity_schema"])>=1; }
        catch(Exception) { return false; }
    }
    private void ChooseAssignments()
    {
        using (AssignmentWindow dialog = new AssignmentWindow(characters, assignments, personGroups, analysisPath != null ? Path.Combine(Path.GetDirectoryName(analysisPath), "assignments.json") : null))
        {
            Forms.DialogResult choice = dialog.ShowDialog(new WindowOwner(this));
            assignments = dialog.Assignments; personGroups = dialog.Groups;
            if (choice != Forms.DialogResult.OK) { stage.Text = "Назначения и группы сохранены до следующего запуска замены."; return; }
            command.Clear();
            StartTask("swap", new Dictionary<string, object> { { "video", videoPath }, { "analysis", analysisPath }, { "assignments", assignments }, { "person_groups", personGroups }, { "excluded_ranges", protectedMoments } });
        }
    }
    private void Execute(string text)
    {
        if (busy || loading || projectPath == null || resultPath == null) return;
        if (String.IsNullOrWhiteSpace(text)) { Error("Опишите, что нужно исправить."); command.Focus(); return; }
        StartTask("execute", new Dictionary<string, object> { { "command", text }, { "project", projectPath } });
    }
    private void Ui(Action action) { if (closed) return; Dispatcher.BeginInvoke(new Action(delegate { if (closed) return; try { action(); } catch (Exception error) { Error(error.Message); } })); }
    private async void StartTask(string action, Dictionary<string, object> job)
    {
        if (busy || loading || closed) return;
        busy = true; activeAction = action; cancelRequested = false; lastError = null; job["action"] = action;
        progress.IsIndeterminate = true; RefreshControls();
        stage.Text = action == "startup" ? "Проверка Ollama…" : action == "last" ? "Открываем результат…" : action == "scan" ? "Ищем людей во всём видео…" : action == "swap" ? "Подготовка замены лиц…" : action=="resume"?"Продолжаем с сохранённой части…":"Разбираем описание…";
        string jobs = Path.Combine(root, "data", "desktop-jobs"); Directory.CreateDirectory(jobs); string input = Path.Combine(jobs, Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(input, Serializer().Serialize(job), new UTF8Encoding(false));
        Process process = new Process(); backend = process; bool received = false, producedVideo = false; StringBuilder errors = new StringBuilder();
        process.StartInfo = new ProcessStartInfo(Path.Combine(root, "runtime", "python", "python.exe"), Args(new string[] { "-u", Path.Combine(root, "desktop_backend.py"), input })) { WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden, RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8 };
        process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
            if (String.IsNullOrWhiteSpace(e.Data)) return;
            try {
                Dictionary<string, object> message = Serializer().Deserialize<Dictionary<string, object>>(e.Data); string type = Convert.ToString(message["type"]);
                if (type != "progress") received = true;
                if (type == "completed" && message.ContainsKey("produced_video")) producedVideo = Convert.ToBoolean(message["produced_video"]);
                Ui(delegate { if (backend == process) Receive(message); });
            } catch (Exception error) { lock (errors) errors.AppendLine(error.Message); }
        };
        process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) lock (errors) errors.AppendLine(e.Data); };
        try
        {
            process.Start(); process.BeginOutputReadLine(); process.BeginErrorReadLine();
            await Task.Run(delegate { while (!process.WaitForExit(1000)) { } Thread.Sleep(200); });
            if (closed || backend != process) return;
            int code = process.ExitCode; string details; lock (errors) details = errors.ToString(); backend = null; busy = false; progress.IsIndeterminate = false; RefreshControls();
            if (cancelRequested) stage.Text = hasPendingRender?"Обработка остановлена. Готовые части сохранены. Нажмите «Продолжить обработку».":"Обработка остановлена. Готовые части поиска сохранены; можно запустить поиск снова.";
            else if (!received || code != 0) { if (lastError == null) Error("Операция не завершена.\n" + details); }
            else if (lastError == null)
            {
                if (producedVideo && resultPath != null && File.Exists(resultPath)) NotifyReady();
                if (action == "startup") StartTask("last", new Dictionary<string, object>());
                else if (action == "scan" && characters.Length > 0) ChooseAssignments();
            }
        }
        catch (Exception error) { if (!closed) { backend = null; busy = false; RefreshControls(); Error(error.Message); } }
        finally { try { process.CancelOutputRead(); } catch (InvalidOperationException) { } try { process.CancelErrorRead(); } catch (InvalidOperationException) { } process.Dispose(); }
    }
    private void Receive(Dictionary<string, object> message)
    {
        string type = Convert.ToString(message["type"]);
        if (type == "progress") { stage.Text = Convert.ToString(message["text"]); progress.IsIndeterminate = message["value"] == null; if (!progress.IsIndeterminate) progress.Value = Math.Max(0, Math.Min(100, Convert.ToDouble(message["value"]) * 100)); }
        else if (type == "notice") { report.Text = Convert.ToString(message["text"]); }
        else if (type == "ready") { localStatus.Text = Convert.ToString(message["text"]); }
        else if (type == "empty") { stage.Text = Convert.ToString(message["text"]); }
        else if (type == "error") Error(Convert.ToString(message["text"]));
        else if (type == "characters") { analysisPath = Convert.ToString(message["analysis"]); analysisVideo = Convert.ToString(message["video"]); characters = Items(message["characters"]); assignments = message.ContainsKey("assignments") ? Items(message["assignments"]) : new object[0]; personGroups = message.ContainsKey("person_groups") ? Items(message["person_groups"]) : new object[0]; stage.Text = "Карточек: " + characters.Length + ". Групп: " + (message.ContainsKey("visible_cards") ? Convert.ToString(message["visible_cards"]) : characters.Length.ToString()); if(message.ContainsKey("identity_summary")&&!String.IsNullOrWhiteSpace(Convert.ToString(message["identity_summary"]))) report.Text=Convert.ToString(message["identity_summary"]); }
        else if(type=="checkpoint") {
            projectPath=Convert.ToString(message["project"]);hasPendingRender=true;RefreshControls();
        }
        else if(type=="pending") {
            projectPath=Convert.ToString(message["project"]);videoPath=Convert.ToString(message["video"]);resultPath=Convert.ToString(message["result"]);
            analysisPath=Convert.ToString(message["analysis"]);analysisVideo=videoPath;
            characters=Items(message["characters"]);assignments=Items(message["assignments"]);personGroups=message.ContainsKey("person_groups")?Items(message["person_groups"]):new object[0];
            protectedMoments=ReadProtection(message["excluded_ranges"]);RenderProtection();hasPendingRender=true;
            stage.Text="Сохранено частей: "+message["completed_parts"]+" / "+message["total_parts"]+". Нажмите «Продолжить обработку».";
            report.Text="Обработка прервана. Готовые минутные части сохранены и будут использованы при продолжении.";
            OpenPreview(videoPath,"Исходник");Task<Dictionary<string,object>> unused=UpdateOriginalInfo(videoPath);RefreshControls();
        }
        else if (type == "completed")
        {
            hasPendingRender=false;
            projectPath = Convert.ToString(message["project"]); resultPath = Convert.ToString(message["result"]); videoPath = Convert.ToString(message["video"]);
            analysisPath = message.ContainsKey("analysis") ? Convert.ToString(message["analysis"]) : null; analysisVideo = videoPath;
            characters = message.ContainsKey("characters") ? Items(message["characters"]) : new object[0]; assignments = message.ContainsKey("assignments") ? Items(message["assignments"]) : new object[0]; personGroups = message.ContainsKey("person_groups") ? Items(message["person_groups"]) : new object[0];
            protectedMoments=message.ContainsKey("excluded_ranges")?ReadProtection(message["excluded_ranges"]):new List<double[]>();RenderProtection();
            SaveProtection(videoPath,protectedMoments);SaveProtection(resultPath,protectedMoments);
            report.Text = Convert.ToString(message["report"]).Replace("**", ""); report.ScrollToHome(); stage.Text = "Готово. Видео и отчёт сохранены.";
            OpenPreview(resultPath, "Результат"); RefreshControls();
            Task<Dictionary<string, object>> unused = UpdateOriginalInfo(videoPath);
        }
    }
    private async Task<Dictionary<string, object>> UpdateOriginalInfo(string path)
    {
        try { Dictionary<string, object> info = await Probe(path); if (videoPath == path && !closed) { videoDuration = Convert.ToDouble(info["duration"]); videoFps = Convert.ToDouble(info["fps"]); } return info; } catch { return null; }
    }
    private void NotifyReady() { ++signalCount; if (diagnosticPath == null) readySound.Play(); }
    private void Error(string text)
    {
        lastError = text; stage.Text = "Операция не завершена.";
        if (text.Contains("Видео изменилось после поиска")) { analysisPath = null; characters = new object[0]; assignments = new object[0]; personGroups = new object[0]; }
        if (diagnosticPath == null) MessageBox.Show(this, text, "FaceStudio", MessageBoxButton.OK, MessageBoxImage.Warning);
    }
    private void CancelTask()
    {
        if (CancelCut()) return;
        Process process = backend; if (process == null) return; cancelRequested = true; RefreshControls(); stage.Text = "Отменяем обработку…";
        Task.Run(delegate { StopOwnedTree(process); });
    }
    private void StopOwnedTree(Process process)
    {
        try { if (!process.HasExited) using (Process stop = Process.Start(new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "taskkill.exe"), "/PID " + process.Id + " /T /F") { UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden })) stop.WaitForExit(10000); } catch (InvalidOperationException) { }
    }
    private void SaveVideo()
    {
        if (resultPath == null) return;
        using (Forms.SaveFileDialog dialog = new Forms.SaveFileDialog { Title = "Сохранить результат", Filter = "Видео MP4|*.mp4", FileName = "FaceStudio-result.mp4" })
        {
            if (dialog.ShowDialog(new WindowOwner(this)) != Forms.DialogResult.OK) return;
            try { if (!String.Equals(Path.GetFullPath(dialog.FileName), Path.GetFullPath(resultPath), StringComparison.OrdinalIgnoreCase)) File.Copy(resultPath, dialog.FileName, true); stage.Text = "Результат сохранён: " + Path.GetFileName(dialog.FileName); } catch (Exception error) { Error(error.Message); }
        }
    }
    private void OnClosing(object sender, CancelEventArgs e)
    {
        if (busy && diagnosticPath == null && MessageBox.Show(this, "Обработка ещё идёт. Отменить её и закрыть программу?", "FaceStudio", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) { e.Cancel = true; return; }
        closed = true; ++loadVersion; ++mediaVersion; timer.Stop(); CloseCutter(); media.Close(); StopPreviewConversion(); if (backend != null) StopOwnedTree(backend); readySound.Dispose();
    }
    private async void RunDiagnostics()
    {
        try
        {
            string fixture = videoPath;
            string spec = diagnosticPath + ".input.json";
            Dictionary<string, object> input = File.Exists(spec) ? Serializer().Deserialize<Dictionary<string, object>>(File.ReadAllText(spec)) : null;
            if (input != null && input.ContainsKey("video")) fixture = Convert.ToString(input["video"]);
            diagnostics["restore_no_signal"] = signalCount == 0;
            diagnostics["restored_result"] = resultPath; diagnostics["folder_button_readable"] = folder.IsEnabled && folder.ActualHeight >= 36 && folder.FontSize >= 12;
            using (AssignmentWindow dialog = new AssignmentWindow(characters, assignments, personGroups)) {
                dialog.Show(); Forms.Application.DoEvents();
                diagnostics["person_profiles"] = dialog.VerifyPersonProfiles();
                if(!Convert.ToBoolean(diagnostics["person_profiles"])) throw new IOException("People profile persistence failed.");
                diagnostics["visible_people"] = dialog.VisibleCardCount;
                diagnostics["assignment_rows"] = dialog.PersonCount; diagnostics["assignment_choices"] = dialog.VerifyChoices(); diagnostics["assignment_groups"] = dialog.VerifyGrouping();
                if(input != null && input.ContainsKey("draft_check_dir")) { diagnostics["draft_persistence"] = dialog.VerifyDraftPersistence(Convert.ToString(input["draft_check_dir"])); if(!Convert.ToBoolean(diagnostics["draft_persistence"])) throw new IOException("Draft persistence check failed."); }
                if (characters.Length >= 3 && personGroups.Length == 0) dialog.DiagnosticMerge(new int[] { Convert.ToInt32(((Dictionary<string,object>)characters[0])["id"]), Convert.ToInt32(((Dictionary<string,object>)characters[2])["id"]) });
                using (System.Drawing.Bitmap preview = new System.Drawing.Bitmap(dialog.Width,dialog.Height)) { dialog.DrawToBitmap(preview,new System.Drawing.Rectangle(0,0,dialog.Width,dialog.Height)); preview.Save(diagnosticPath+".groups.png"); }
                dialog.Close();
            }
            diagnosticSelection = fixture; open.RaiseEvent(new RoutedEventArgs(Button.ClickEvent)); while (loading) await Task.Delay(100); await WaitMedia();
            diagnostics["button_loaded"] = opened && videoPath == Path.GetFullPath(fixture) && Convert.ToString(diagnostics["last_load_method"]) == "button";
            diagnostics["button_preview"] = media.NaturalVideoWidth > 0;
            diagnostics["file_name_shown"] = fileName.Text.Contains(Path.GetFileName(fixture));
            diagnostics["duration_shown"] = fileDetails.Text.Contains(Stamp(TimeSpan.FromSeconds(videoDuration)));
            int tasksBefore = signalCount;
            DataObject data = new DataObject(DataFormats.FileDrop, new string[] { fixture });
            DragEventArgs over = DiagnosticDragArgs(data,DragDrop.PreviewDragOverEvent);
            dropArea.RaiseEvent(over); diagnostics["drag_highlight"] = dropHighlight.Visibility == Visibility.Visible && over.Effects == DragDropEffects.Copy;
            DragEventArgs drop = DiagnosticDragArgs(data,DragDrop.PreviewDropEvent);
            dropArea.RaiseEvent(drop); while (loading) await Task.Delay(100); await WaitMedia();
            diagnostics["drop_loaded"] = opened && Convert.ToString(diagnostics["last_load_method"]) == "drop";
            diagnostics["drop_clears_highlight"] = dropHighlight.Visibility != Visibility.Visible;
            diagnostics["load_does_not_swap"] = !busy && backend == null && signalCount == tasksBefore && projectPath == null;
            seek.Value = 500; await Task.Delay(500); diagnostics["seek_works"] = Math.Abs(media.Position.TotalSeconds-videoDuration/2)<.4;
            string preserved = videoPath;
            bool invalid = await OpenDroppedFiles(new string[] { Path.Combine(root,"README.md") });
            diagnostics["invalid_file_rejected"] = !invalid && videoPath == preserved && lastError != null; lastError = null;
            diagnostics["multiple_files_rejected"] = !await OpenDroppedFiles(new string[] { fixture,fixture }); lastError = null;
            busy = true; activeAction = "swap"; RefreshControls();
            diagnostics["busy_drop_rejected"] = !await OpenDroppedFiles(new string[] { fixture }) && videoPath == preserved;
            diagnostics["cancel_available"] = cancel.IsEnabled; busy = false; RefreshControls();
            double oldWidth = Width,oldHeight = Height; Width = MinWidth; Height = MinHeight; UpdateLayout();
            diagnostics["small_layout"] = LayoutFits(); Width = 1560; Height = 960; UpdateLayout(); diagnostics["large_layout"] = LayoutFits(); Width = oldWidth; Height = oldHeight; UpdateLayout();
            diagnostics["native_window"] = true; diagnostics["uses_browser"] = false; diagnostics["command_is_empty"] = command.Text.Length == 0;
            diagnostics["media_opened"] = opened; diagnostics["video_width"] = media.NaturalVideoWidth; diagnostics["video_height"] = media.NaturalVideoHeight;
            diagnostics["has_audio"] = media.HasAudio; diagnostics["duration"] = videoDuration; diagnostics["error"] = lastError;
            bool ok = true; foreach (string key in new string[] { "assignment_choices","assignment_groups","folder_button_readable","restore_no_signal","button_loaded","button_preview","file_name_shown","duration_shown","drag_highlight","drop_loaded","drop_clears_highlight","load_does_not_swap","seek_works","invalid_file_rejected","multiple_files_rejected","busy_drop_rejected","cancel_available","small_layout","large_layout" }) ok &= Convert.ToBoolean(diagnostics[key]);
            if (input != null && input.ContainsKey("alternate_video"))
            {
                bool alternate = await OpenDroppedFiles(new string[] { Convert.ToString(input["alternate_video"]) }); await WaitMedia();
                diagnostics["alternate_format_preview"] = alternate && opened; ok &= alternate && opened;
            }
            if (input != null && input.ContainsKey("job"))
            {
                Dictionary<string,object> job = (Dictionary<string,object>)input["job"];
                StartTask(Convert.ToString(job["action"]),job); while (busy) await Task.Delay(100); await WaitMedia();
                diagnostics["video_completion_signal"] = signalCount == 1 && lastError == null && File.Exists(resultPath); ok &= Convert.ToBoolean(diagnostics["video_completion_signal"]);
                string readyVideo = resultPath; StartTask("check",new Dictionary<string,object> { { "project",projectPath } }); while (busy) await Task.Delay(100); await WaitMedia();
                diagnostics["technical_check_no_signal"] = signalCount == 1 && resultPath == readyVideo && lastError == null; ok &= Convert.ToBoolean(diagnostics["technical_check_no_signal"]);
                diagnostics["result"] = resultPath;
            }
            if(input != null && input.ContainsKey("cut_test")) { ok &= await DiagnoseCutter(Path.GetDirectoryName(diagnosticPath)); }
            if(input!=null&&input.ContainsKey("resume_test")) {
                Receive(new Dictionary<string,object>{{"type","checkpoint"},{"project",diagnosticPath+".project.json"},{"completed_parts",1},{"total_parts",3}});
                bool ready=hasPendingRender&&Find<Button>("ResumeButton").IsEnabled;
                busy=true;activeAction="resume";RefreshControls();bool cancellable=cancel.IsEnabled&&!Find<Button>("ResumeButton").IsEnabled;
                busy=false;RefreshControls();
                Receive(new Dictionary<string,object>{{"type","pending"},{"project",diagnosticPath+".project.json"},{"video",videoPath},{"result",null},{"analysis",null},{"characters",new object[0]},{"assignments",new object[0]},{"excluded_ranges",protectedMoments},{"completed_parts",1},{"total_parts",3}});
                await WaitMedia();diagnostics["resume_controls"]=ready&&cancellable&&opened&&hasPendingRender&&Find<Button>("ResumeButton").IsEnabled;ok &= Convert.ToBoolean(diagnostics["resume_controls"]);
            }
            UpdateLayout();
            FrameworkElement visual = (FrameworkElement)Content;
            System.Windows.Media.Imaging.RenderTargetBitmap bitmap = new System.Windows.Media.Imaging.RenderTargetBitmap((int)(visual.ActualWidth+visual.Margin.Left+visual.Margin.Right),(int)(visual.ActualHeight+visual.Margin.Top+visual.Margin.Bottom),96,96,PixelFormats.Pbgra32);
            DrawingVisual backdrop = new DrawingVisual(); using (DrawingContext drawing = backdrop.RenderOpen()) drawing.DrawRectangle(Background,null,new Rect(0,0,bitmap.PixelWidth,bitmap.PixelHeight));
            bitmap.Render(backdrop); bitmap.Render((Visual)Content); System.Windows.Media.Imaging.PngBitmapEncoder encoder = new System.Windows.Media.Imaging.PngBitmapEncoder(); encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bitmap));
            using (FileStream snapshot = File.Create(diagnosticPath + ".png")) encoder.Save(snapshot);
            diagnostics["ok"] = ok; File.WriteAllText(diagnosticPath, Serializer().Serialize(diagnostics), new UTF8Encoding(false)); Environment.ExitCode = ok ? 0 : 1; Close();
        }
        catch (Exception error) { diagnostics["ok"] = false; diagnostics["error"] = error.ToString(); File.WriteAllText(diagnosticPath, Serializer().Serialize(diagnostics), new UTF8Encoding(false)); Environment.ExitCode = 1; Close(); }
    }
    private bool LayoutFits()
    {
        Point bottom = cancel.TransformToAncestor((Visual)Content).Transform(new Point(0,cancel.ActualHeight));
        foreach(string name in new[]{"CutStart","CutEnd","CutSave","CutUse"}) {
            FrameworkElement item=Find<FrameworkElement>(name);Point top=item.TransformToAncestor((Visual)Content).Transform(new Point());Point edge=item.TransformToAncestor((Visual)Content).Transform(new Point(item.ActualWidth,item.ActualHeight));
            if(top.Y<0||edge.Y>((FrameworkElement)Content).ActualHeight||edge.X>((FrameworkElement)Content).ActualWidth||item.ActualWidth<70)return false;
        }
        return open.ActualWidth >= 100 && swap.ActualWidth >= 235 && repair.ActualWidth >= 235 && command.ActualWidth >= 235 && dropArea.ActualWidth >= 400 && Find<ScrollViewer>("ControlScroll").ActualHeight > 200 && bottom.Y <= ((FrameworkElement)Content).ActualHeight;
    }
    private DragEventArgs DiagnosticDragArgs(DataObject data,RoutedEvent routedEvent)
    {
        DragEventArgs args = (DragEventArgs)Activator.CreateInstance(typeof(DragEventArgs),BindingFlags.Instance | BindingFlags.NonPublic,null,
            new object[] { data,DragDropKeyStates.None,DragDropEffects.Copy,dropArea,new Point(30,30) },CultureInfo.InvariantCulture);
        args.RoutedEvent = routedEvent; return args;
    }
    private async Task WaitMedia() { Stopwatch watch = Stopwatch.StartNew(); while (!opened && watch.Elapsed.TotalSeconds < 30 && !closed) await Task.Delay(100); }
}
internal sealed class WindowOwner : Forms.IWin32Window
{
    private readonly Window window; internal WindowOwner(Window window) { this.window = window; }
    public IntPtr Handle { get { return new WindowInteropHelper(window).Handle; } }
}
