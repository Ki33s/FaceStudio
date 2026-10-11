using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Windows;
using System.Windows.Controls;

internal sealed partial class StudioWindow
{
    private List<double[]> protectedMoments = new List<double[]>();
    private ListBox protectedList;
    private Button protectAdd, protectRemove;
    private TextBlock protectNote;

    private void InitializeProtection()
    {
        protectedList=Find<ListBox>("ProtectedList");
        protectAdd=Find<Button>("ProtectAdd");protectRemove=Find<Button>("ProtectRemove");
        protectNote=Find<TextBlock>("ProtectNote");
        protectAdd.Click+=delegate { if(ReadCutTimes(true))AddProtectedMoment(cutStart,cutEnd); };
        protectRemove.Click+=delegate {
            if(protectedList.SelectedIndex<0)return;
            protectedMoments.RemoveAt(protectedList.SelectedIndex);ProtectionChanged();
        };
        protectedList.SelectionChanged+=delegate { RefreshProtectionControls(); };
        protectedList.MouseDoubleClick+=delegate {
            int i=protectedList.SelectedIndex;if(i<0||busy||loading)return;
            cutStart=protectedMoments[i][0];cutEnd=Math.Min(cutDuration,protectedMoments[i][1]);SyncCut();SeekCut(cutStart);
        };
        RenderProtection();
    }
    private static List<double[]> MergeProtection(List<double[]> ranges)
    {
        ranges.Sort(delegate(double[] a,double[] b){return a[0].CompareTo(b[0]);});
        List<double[]> merged=new List<double[]>();
        foreach(double[] range in ranges){
            if(merged.Count>0&&range[0]<=merged[merged.Count-1][1])merged[merged.Count-1][1]=Math.Max(merged[merged.Count-1][1],range[1]);
            else merged.Add(new[]{range[0],range[1]});
        }
        return merged;
    }
    private void AddProtectedMoment(double a,double b)
    {
        if(busy||loading||viewKind=="Результат"||b<=a)return;
        protectedMoments.Add(new[]{a,b});protectedMoments=MergeProtection(protectedMoments);ProtectionChanged();
        stage.Text="Момент отмечен: лица останутся исходными. Теперь выберите фрагмент для обрезки или отметьте ещё один момент.";
    }
    private void ProtectionChanged(){RenderProtection();if(videoPath!=null)SaveProtection(videoPath,protectedMoments);}
    private void RenderProtection()
    {
        if(protectedList==null)return;
        protectedList.Items.Clear();foreach(double[] range in protectedMoments)protectedList.Items.Add(CutStamp(range[0])+" — "+CutStamp(range[1]));
        protectedList.Visibility=protectedMoments.Count==0?Visibility.Collapsed:Visibility.Visible;
        cutTimeline.ProtectedRanges=protectedMoments;cutTimeline.InvalidateVisual();RefreshProtectionControls();
    }
    private void RefreshProtectionControls()
    {
        if(protectAdd==null)return;
        bool available=!busy&&!loading&&cutSource!=null&&cutDuration>0&&viewKind!="Результат";
        protectAdd.IsEnabled=available;protectRemove.IsEnabled=available&&protectedList.SelectedIndex>=0;
        protectNote.Text=viewKind=="Результат"?"Отметки проекта. Для новой замены изменяйте их в «Исходнике».":"Выделите момент на шкале и добавьте. Затем выберите границы обрезки. Отметки применяются при следующей замене лиц.";
        cutMode.IsEnabled=!busy&&!loading&&cutSource!=null&&cutDuration>0&&protectedMoments.Count==0;
        if(protectedMoments.Count>0){cutMode.SelectedIndex=0;Find<TextBlock>("CutModeNote").Text="Точная обрезка сохраняет время отметок без замены лиц.";}
        else Find<TextBlock>("CutModeNote").Text=cutMode.SelectedIndex==0?"Точная обрезка, высокое качество.":"Без перекодирования. Начало может сместиться к ключевому кадру.";
    }
    private string ProtectionFile(string path)
    {
        FileInfo info=new FileInfo(path);
        string identity=info.FullName.ToUpperInvariant()+"|"+info.Length+"|"+info.LastWriteTimeUtc.Ticks;
        using(SHA256 hash=SHA256.Create())return Path.Combine(root,"data","edit-marks",BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(identity))).Replace("-","")+".json");
    }
    private void SaveProtection(string path,List<double[]> ranges)
    {
        try{
            string target=ProtectionFile(path);Directory.CreateDirectory(Path.GetDirectoryName(target));
            string temp=target+"."+Guid.NewGuid().ToString("N")+".tmp";
            File.WriteAllText(temp,Serializer().Serialize(ranges),new UTF8Encoding(false));
            if(File.Exists(target))File.Replace(temp,target,null);else File.Move(temp,target);
        }catch(Exception error){Error("Не удалось сохранить отметки без замены лиц: "+error.Message);}
    }
    private List<double[]> ReadProtection(object values)
    {
        List<double[]> ranges=new List<double[]>();
        foreach(object value in Items(values)){
            object[] pair=Items(value);if(pair.Length!=2)continue;
            double a=Convert.ToDouble(pair[0]),b=Convert.ToDouble(pair[1]);
            if(!double.IsNaN(a)&&!double.IsNaN(b)&&!double.IsInfinity(a)&&!double.IsInfinity(b)&&a>=0&&b>a)ranges.Add(new[]{a,b});
        }
        return MergeProtection(ranges);
    }
    private void LoadProtection(string path)
    {
        protectedMoments=new List<double[]>();
        try{string file=ProtectionFile(path);if(File.Exists(file))protectedMoments=ReadProtection(Serializer().DeserializeObject(File.ReadAllText(file)));}catch{ }
        RenderProtection();
    }
    private List<double[]> CroppedProtection(double start,double end)
    {
        List<double[]> ranges=new List<double[]>();
        foreach(double[] range in protectedMoments){double a=Math.Max(start,range[0]),b=Math.Min(end,range[1]);if(b>a)ranges.Add(new[]{a-start,b-start});}
        return ranges;
    }
}
