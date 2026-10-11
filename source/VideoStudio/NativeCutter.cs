using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Forms = System.Windows.Forms;

internal sealed partial class StudioWindow
{
    private readonly Cutter.RangeTimeline cutTimeline = new Cutter.RangeTimeline();
    private TextBox cutStartField, cutEndField;
    private Button cutSave, cutUse, cutPreview, cutReset, cutMarkStart, cutMarkEnd;
    private ComboBox cutMode;
    private TextBlock cutLength, cutSourceLabel;
    private double cutStart, cutEnd, cutDuration;
    private string cutSource, cutStaging;
    private bool updatingCut, cutPreviewing;
    private Process cutProcess;

    private void InitializeCutter()
    {
        cutStartField = Find<TextBox>("CutStart"); cutEndField = Find<TextBox>("CutEnd");
        cutSave = Find<Button>("CutSave"); cutUse = Find<Button>("CutUse"); cutPreview = Find<Button>("CutPreview"); cutReset = Find<Button>("CutReset");
        cutMarkStart = Find<Button>("CutMarkStart"); cutMarkEnd = Find<Button>("CutMarkEnd");
        cutMode = Find<ComboBox>("CutMode"); cutLength = Find<TextBlock>("CutLength"); cutSourceLabel = Find<TextBlock>("CutSourceLabel");
        Find<Border>("CutTimelineHost").Child = cutTimeline;
        cutTimeline.IsEnabled = false;
        cutTimeline.BoundaryChanged = delegate(bool first,double time) { PauseCutPlayer(); SetCutBoundary(first,time); SeekCut(first ? cutStart : cutEnd); };
        cutTimeline.SeekRequested = delegate(double time) { cutPreviewing = false; SeekCut(time); };
        cutMarkStart.Click += delegate { SetCutBoundary(true,media.Position.TotalSeconds); };
        cutMarkEnd.Click += delegate { SetCutBoundary(false,media.Position.TotalSeconds); };
        cutReset.Click += delegate { cutStart=0;cutEnd=cutDuration;cutPreviewing=false;SyncCut(); };
        cutPreview.Click += delegate { if(ReadCutTimes(true) && opened) {SeekCut(cutStart);cutPreviewing=true;media.Play();paused=false;play.Content="Ⅱ";} };
        cutSave.Click += async delegate { await SaveCut(); };
        cutUse.Click += async delegate { await UseCut(); };
        foreach(TextBox field in new [] {cutStartField,cutEndField}) {
            field.TextChanged += delegate { if(!updatingCut)ReadCutTimes(false); };
            field.LostKeyboardFocus += delegate { if(!updatingCut && ReadCutTimes(false))SyncCut(); };
            field.KeyDown += delegate(object s,KeyEventArgs e) { if(e.Key==Key.Enter){ReadCutTimes(true);e.Handled=true;} };
        }
        cutMode.SelectionChanged += delegate { Find<TextBlock>("CutModeNote").Text = cutMode.SelectedIndex==0 ? "Точная обрезка, высокое качество." : "Без перекодирования. Начало может сместиться к ключевому кадру."; };
        Find<Button>("FrameBack").Click += delegate { PauseCutPlayer();SeekCut(media.Position.TotalSeconds-1/videoFps); };
        Find<Button>("FrameNext").Click += delegate { PauseCutPlayer();SeekCut(media.Position.TotalSeconds+1/videoFps); };
        PreviewKeyDown += delegate(object s,KeyEventArgs e) {
            if(busy||loading||Keyboard.FocusedElement is TextBox||Keyboard.FocusedElement is ComboBox||Keyboard.FocusedElement is Cutter.RangeTimeline)return;
            if((Keyboard.Modifiers&ModifierKeys.Control)!=0 && e.Key==Key.O){Task unused=BrowseVideo();e.Handled=true;}
            else if(e.Key==Key.I){SetCutBoundary(true,media.Position.TotalSeconds);e.Handled=true;}
            else if(e.Key==Key.O){SetCutBoundary(false,media.Position.TotalSeconds);e.Handled=true;}
        };
        InitializeProtection();RefreshCutControls();
    }
    private static string CutStamp(double seconds)
    {
        TimeSpan t=TimeSpan.FromMilliseconds(Math.Max(0,Math.Round(seconds*1000)));
        return t.TotalHours>=1?t.ToString(@"hh\:mm\:ss\.fff"):t.ToString(@"mm\:ss\.fff");
    }
    private static double ParseCutTime(string text)
    {
        string[] parts=text.Trim().Replace(',','.').Split(':');if(parts.Length>3)return double.NaN;
        double value=0;foreach(string part in parts){double n;if(!double.TryParse(part,NumberStyles.AllowDecimalPoint,CultureInfo.InvariantCulture,out n)||n<0)return double.NaN;value=value*60+n;}return value;
    }
    private bool ReadCutTimes(bool notify)
    {
        if(cutDuration<=0)return false;
        double a=ParseCutTime(cutStartField.Text),b=ParseCutTime(cutEndField.Text);
        if(double.IsNaN(a)||double.IsNaN(b)||double.IsInfinity(a)||double.IsInfinity(b)||a<0||b>cutDuration+.001||b-a<.01){if(notify)Error("Проверьте начало и конец фрагмента. Конец должен быть позже начала и не выходить за длину видео.");return false;}
        cutStart=a;cutEnd=Math.Min(b,cutDuration);RenderCut();return true;
    }
    private void RenderCut(){cutLength.Text=CutStamp(cutEnd-cutStart);cutTimeline.StartTime=cutStart;cutTimeline.EndTime=cutEnd;cutTimeline.InvalidateVisual();}
    private void SyncCut(){updatingCut=true;cutStartField.Text=CutStamp(cutStart);cutEndField.Text=CutStamp(cutEnd);updatingCut=false;RenderCut();}
    private void SetCutBoundary(bool first,double time){if(busy||loading||cutDuration<=0)return;cutPreviewing=false;if(first)cutStart=Math.Max(0,Math.Min(time,cutEnd-.01));else cutEnd=Math.Min(cutDuration,Math.Max(time,cutStart+.01));SyncCut();}
    private void SeekCut(double time){if(!opened||cutDuration<=0)return;media.Position=TimeSpan.FromSeconds(Math.Max(0,Math.Min(cutDuration,time)));UpdateTime();}
    private void PauseCutPlayer(){media.Pause();paused=true;play.Content="▶";cutPreviewing=false;}
    private async Task PrepareCut(string path,int version)
    {
        cutSource=path;cutDuration=0;cutPreviewing=false;RefreshCutControls();cutSourceLabel.Text="Обрезка: "+viewKind.ToLowerInvariant();
        try{Dictionary<string,object> metadata=await Probe(path);if(closed||version!=mediaVersion)return;cutDuration=Convert.ToDouble(metadata["duration"]);cutStart=0;cutEnd=cutDuration;cutTimeline.Duration=cutDuration;cutTimeline.Position=0;SyncCut();RefreshCutControls();}
        catch{if(!closed&&version==mediaVersion){cutSource=null;cutSourceLabel.Text="Обрезка недоступна для этого файла";RefreshCutControls();}}
    }
    private void RefreshCutControls()
    {
        if(cutSave==null)return;
        bool available=!busy&&!loading&&cutSource!=null&&cutDuration>0;
        cutStartField.IsEnabled=cutEndField.IsEnabled=cutSave.IsEnabled=cutUse.IsEnabled=cutReset.IsEnabled=cutMarkStart.IsEnabled=cutMarkEnd.IsEnabled=cutTimeline.IsEnabled=cutMode.IsEnabled=available;
        cutPreview.IsEnabled=available&&opened;RefreshProtectionControls();
        Find<Button>("FrameBack").IsEnabled=Find<Button>("FrameNext").IsEnabled=opened;
        if(cutProcess!=null){cancel.IsEnabled=!cancelRequested;sourceButton.IsEnabled=resultButton.IsEnabled=false;}
    }
    private void TickCutter()
    {
        if(!opened||cutDuration<=0)return;
        cutTimeline.Position=media.Position.TotalSeconds;cutTimeline.InvalidateVisual();
        position.Text=CutStamp(media.Position.TotalSeconds)+" / "+CutStamp(cutDuration);
        if(cutPreviewing&&media.Position.TotalSeconds>=cutEnd){PauseCutPlayer();media.Position=TimeSpan.FromSeconds(cutEnd);}
    }
    private async Task SaveCut()
    {
        if(busy||loading||!ReadCutTimes(true))return;
        using(Forms.SaveFileDialog dialog=new Forms.SaveFileDialog{Title="Сохранить фрагмент",Filter="Видео MP4|*.mp4",DefaultExt="mp4",AddExtension=true,OverwritePrompt=true,FileName=Path.GetFileNameWithoutExtension(cutSource)+"_фрагмент.mp4"}){
            if(dialog.ShowDialog(new WindowOwner(this))!=Forms.DialogResult.OK)return;
            if(await ExportCut(dialog.FileName,cutMode.SelectedIndex==1)){stage.Text="Фрагмент сохранён: "+dialog.FileName;}
        }
    }
    private async Task UseCut()
    {
        if(busy||loading||!ReadCutTimes(true))return;
        string clips=Path.Combine(root,"data","clips");Directory.CreateDirectory(clips);
        string output=Path.Combine(clips,Path.GetFileNameWithoutExtension(cutSource)+"_"+DateTime.Now.ToString("yyyyMMdd-HHmmss")+"_"+Guid.NewGuid().ToString("N").Substring(0,6)+".mp4");
        if(await ExportCut(output,false)){
            if(await OpenVideo(output,"cut")){
                report.Text="Фрагмент готов для замены лиц. Нажмите «Заменить лица», чтобы найти людей и выбрать фотографии.\n\nФайл фрагмента: "+output;
                stage.Text="Фрагмент выбран. Теперь нажмите «Заменить лица».";
            }
        }
    }
    private string[] CutArguments(string source,string output,bool fast)
    {
        List<string> args=new List<string>{"-v","error","-ss",cutStart.ToString("0.########",CultureInfo.InvariantCulture),"-i",source,"-t",(cutEnd-cutStart).ToString("0.########",CultureInfo.InvariantCulture),"-map","0:v:0","-map","0:a:0?"};
        if(fast)args.AddRange(new []{"-c","copy","-avoid_negative_ts","make_zero"});
        else args.AddRange(new []{"-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k"});
        args.AddRange(new []{"-movflags","+faststart","-progress","pipe:1","-nostats","-y",output});return args.ToArray();
    }
    private async Task<bool> ExportCut(string destination,bool fast)
    {
        if(busy||loading||!ReadCutTimes(true))return false;
        string source=cutSource,target=Path.GetFullPath(destination);
        List<double[]> marks=CroppedProtection(cutStart,cutEnd);if(protectedMoments.Count>0)fast=false;
        if(String.Equals(source,target,StringComparison.OrdinalIgnoreCase)){Error("Выберите другое имя. Исходный файл перезаписывать нельзя.");return false;}
        PauseCutPlayer();busy=true;activeAction="cut";cancelRequested=false;lastError=null;
        string[] arguments;double duration=cutEnd-cutStart;
        cutStaging=Path.Combine(Path.GetDirectoryName(target),".videostudio-"+Guid.NewGuid().ToString("N")+".mp4");arguments=CutArguments(source,cutStaging,fast);
        progress.IsIndeterminate=false;progress.Value=0;stage.Text="Сохраняем фрагмент…";
        Process process=new Process();cutProcess=process;RefreshControls();
        try{
            process.StartInfo=new ProcessStartInfo(Path.Combine(root,"runtime","bin","ffmpeg.exe"),Args(arguments)){UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true,StandardOutputEncoding=Encoding.UTF8,StandardErrorEncoding=Encoding.UTF8};process.Start();
            Task<string> errors=process.StandardError.ReadToEndAsync();
            Task output=Task.Run(async delegate{string line;while((line=await process.StandardOutput.ReadLineAsync())!=null){if(line.StartsWith("out_time_us=")){long us;if(long.TryParse(line.Substring(12),out us)){double percent=Math.Min(99,Math.Max(0,us/1000000.0/duration*100));Ui(delegate{if(cutProcess==process){progress.Value=percent;stage.Text="Сохраняем фрагмент… "+Math.Round(percent)+"%";}});}}}});
            await Task.Run(delegate{process.WaitForExit();});await output;string details=await errors;
            if(cancelRequested||closed)return false;
            if(process.ExitCode!=0)throw new IOException(details);
            await Probe(cutStaging);if(cancelRequested||closed)return false;
            if(File.Exists(target))File.Replace(cutStaging,target,null);else File.Move(cutStaging,target);
            SaveProtection(target,marks);progress.Value=100;stage.Text="Фрагмент готов.";return true;
        }catch(Exception error){if(!closed&&!cancelRequested)Error(fast?"Быстрый режим не подходит для этого файла. Выберите «Точно».":"Не удалось сохранить фрагмент: "+error.Message);return false;}
        finally{cutProcess=null;process.Dispose();try{if(File.Exists(cutStaging))File.Delete(cutStaging);}catch{}cutStaging=null;busy=false;if(!closed){if(cancelRequested)stage.Text="Сохранение фрагмента отменено.";RefreshControls();}}
    }
    private bool CancelCut()
    {
        Process process=cutProcess;if(process==null)return false;cancelRequested=true;stage.Text="Отменяем сохранение фрагмента…";
        try{if(!process.HasExited){process.Kill();process.WaitForExit(2000);}}catch(InvalidOperationException){}RefreshControls();return true;
    }
    private void CloseCutter(){CancelCut();try{if(File.Exists(cutStaging))File.Delete(cutStaging);}catch{}}
    private async Task<bool> DiagnoseCutter(string folder)
    {
        while(cutDuration<=0)await Task.Delay(100);
        protectedMoments.Clear();RenderProtection();
        cutStartField.Text="00:00.800";cutEndField.Text="00:02.000";protectAdd.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        cutStartField.Text="00:01.800";cutEndField.Text="00:02.800";protectAdd.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        cutStartField.Text="00:04.000";cutEndField.Text="00:05.000";protectAdd.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        diagnostics["protection_merge"]=protectedMoments.Count==2&&Math.Abs(protectedMoments[0][0]-.8)<.001&&Math.Abs(protectedMoments[0][1]-2.8)<.001;
        cutStartField.Text="00:05.500";cutEndField.Text="00:06.000";protectAdd.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        protectedList.SelectedIndex=2;protectRemove.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        diagnostics["protection_remove"]=protectedMoments.Count==2;
        LoadProtection(videoPath);diagnostics["protection_persistence"]=protectedMoments.Count==2;
        cutStartField.Text="00:01.320";cutEndField.Text="00:03.720";
        bool controls=ReadCutTimes(true)&&Math.Abs(cutEnd-cutStart-2.4)<.001&&cutSave.IsEnabled&&cutUse.IsEnabled;
        string output=Path.Combine(folder,"integrated-fragment.mp4");
        bool exported=await ExportCut(output,false);var metadata=exported?await Probe(output):null;
        bool duration=metadata!=null&&Math.Abs(Convert.ToDouble(metadata["duration"])-2.4)<Math.Max(.09,1/videoFps+.02);
        bool used=exported&&await OpenVideo(output,"cut");await WaitMedia();
        while(used&&cutDuration<=0)await Task.Delay(100);
        diagnostics["protection_crop_shift"]=protectedMoments.Count==1&&Math.Abs(protectedMoments[0][0])<.001&&Math.Abs(protectedMoments[0][1]-1.48)<.001;
        bool bridge=used&&videoPath==output&&projectPath==null&&analysisPath==null&&swap.IsEnabled&&Convert.ToString(diagnostics["last_load_method"])=="cut";
        diagnostics["cut_controls"]=controls;diagnostics["cut_export"]=exported;diagnostics["cut_duration"]=duration;diagnostics["cut_to_faces"]=bridge;
        if(bridge){OpenPreview(output,"Результат");await WaitMedia();while(cutDuration<=0)await Task.Delay(100);diagnostics["cut_result_view"]=cutSource==output&&cutSourceLabel.Text.Contains("результат");}else diagnostics["cut_result_view"]=false;
        diagnostics["protection_result_kept"]=protectedMoments.Count==1&&Math.Abs(protectedMoments[0][1]-1.48)<.001&&!protectAdd.IsEnabled;
        bool marksOk=true;foreach(string key in new[]{"protection_merge","protection_remove","protection_persistence","protection_crop_shift","protection_result_kept"})marksOk &= Convert.ToBoolean(diagnostics[key]);
        return marksOk&&controls&&exported&&duration&&bridge&&Convert.ToBoolean(diagnostics["cut_result_view"]);
    }
}

