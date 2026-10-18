using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Security.Cryptography;
using System.Windows.Forms;
using System.Web.Script.Serialization;

internal sealed class AssignmentWindow : Form
{
    private sealed class Photo
    {
        internal string Primary;
        internal List<string> Extra = new List<string>();
        internal Photo Copy() { return new Photo { Primary=Primary, Extra=new List<string>(Extra) }; }
    }
    private sealed class PersonRow
    {
        internal int Id;
        internal string Thumbnail, PersonUid;
        internal List<string> Examples = new List<string>();
        internal double First, Last;
        internal Photo Photo = new Photo();
        internal TableLayoutPanel Card;
        internal CheckBox Selected;
        internal PictureBox Original, Replacement;
        internal Label Caption, FileName;
        internal Button Additional, Clear;
    }
    private readonly object[] people;
    private readonly List<PersonRow> rows = new List<PersonRow>();
    private readonly List<List<PersonRow>> groups = new List<List<PersonRow>>();
    private readonly TableLayoutPanel list = new TableLayoutPanel { Dock=DockStyle.Top,AutoSize=true,ColumnCount=1 };
    private readonly Button start = new RoundedButton(), merge, split, selectedPhoto, unselect;
    private readonly Label count = new Label();
    private readonly ToolTip tips = new ToolTip();
    private readonly Dictionary<string,string> photoHashes = new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
    private bool updating;
    private readonly string draftPath;
    private bool draftReady;
    private string draftFingerprint, draftError;
    internal int PersonCount { get { return rows.Count; } }
    internal int VisibleCardCount { get { return groups.Count; } }
    internal bool CanStart { get { return start.Enabled; } }
    internal object[] Groups
    {
        get {
            List<object> result=new List<object>();
            foreach (List<PersonRow> group in groups) if (group.Count>1) {
                List<int> ids=new List<int>();foreach (PersonRow row in group) ids.Add(row.Id);result.Add(ids.ToArray());
            }
            return result.ToArray();
        }
    }
    internal object[] Assignments
    {
        get {
            List<object> result=new List<object>();
            foreach (PersonRow row in rows) if (File.Exists(row.Photo.Primary)) {
                List<string> paths=new List<string>();paths.Add(row.Photo.Primary);
                foreach (string path in row.Photo.Extra) if (File.Exists(path)) paths.Add(path);
                result.Add(new Dictionary<string,object>{{"person_id",row.Id},{"sources",paths.ToArray()}});
            }
            return result.ToArray();
        }
    }
    internal AssignmentWindow(object[] people,object[] saved) : this(people,saved,new object[0]) { }
    internal AssignmentWindow(object[] people,object[] saved,object[] savedGroups) : this(people,saved,savedGroups,null) { }
    internal AssignmentWindow(object[] people,object[] saved,object[] savedGroups,string draft)
    {
        this.people=people;draftPath=draft;
        if(draftPath!=null) {
            JavaScriptSerializer serializer=new JavaScriptSerializer { MaxJsonLength=Int32.MaxValue };
            string analysis=Path.Combine(Path.GetDirectoryName(draftPath),"people.json");
            if(File.Exists(analysis)) {
                Dictionary<string,object> info=serializer.Deserialize<Dictionary<string,object>>(File.ReadAllText(analysis));
                draftFingerprint=Convert.ToString(info.ContainsKey("input_fingerprint")?info["input_fingerprint"]:info["fingerprint"]);
            }
            if(saved.Length==0&&File.Exists(draftPath)) {
                Dictionary<string,object> value=serializer.Deserialize<Dictionary<string,object>>(File.ReadAllText(draftPath));
                HashSet<int> known=new HashSet<int>(),stored=new HashSet<int>();
                foreach(object item in people)known.Add(Convert.ToInt32(((Dictionary<string,object>)item)["id"]));
                if(value.ContainsKey("person_ids"))foreach(object id in (System.Collections.IEnumerable)value["person_ids"])stored.Add(Convert.ToInt32(id));
                if(known.SetEquals(stored)&&value.ContainsKey("input_fingerprint")&&Convert.ToString(value["input_fingerprint"])==draftFingerprint) {
                    saved=Objects(value["assignments"]);savedGroups=Objects(value["person_groups"]);
                }
            }
        }
        Text="Кого заменить — FaceStudio";Font=new Font("Segoe UI",10);
        StartPosition=FormStartPosition.CenterParent;Size=new Size(1010,780);MinimumSize=new Size(900,580);
        BackColor=Color.FromArgb(17,21,18);ForeColor=Color.FromArgb(239,243,233);AutoScaleMode=AutoScaleMode.Dpi;
        TableLayoutPanel layout=new TableLayoutPanel { Dock=DockStyle.Fill,Padding=new Padding(18),ColumnCount=1,RowCount=4 };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute,60));layout.RowStyles.Add(new RowStyle(SizeType.Absolute,84));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.Absolute,56));
        layout.Controls.Add(new Label { Text="Одно фото применяется ко всем появлениям человека. Дорожки проверены вперёд и назад, сцены сопоставлены по всему ролику. Ошибочную группу можно разделить. Назначения сохраняются автоматически.",Dock=DockStyle.Fill,TextAlign=ContentAlignment.MiddleLeft },0,0);
        FlowLayoutPanel top=new FlowLayoutPanel { Dock=DockStyle.Fill,WrapContents=true };
        merge=MakeButton("Это один человек",185);merge.Click+=delegate { MergeSelected(null,true); };
        selectedPhoto=MakeButton("Фото выбранным",175);selectedPhoto.Click+=delegate { string path=PickPhoto();if(path==null)return;foreach(List<PersonRow> group in SelectedGroups()) ApplyPhoto(group,new Photo { Primary=path });UpdateCount(); };
        split=MakeButton("Разделить группу",175);split.Click+=delegate { SplitSelected(); };
        unselect=MakeButton("Снять выделение",160);unselect.Click+=delegate { ClearSelection();UpdateCount(); };
        Button all=MakeButton("Одно фото всем",170);all.Click+=delegate { string path=PickPhoto();if(path==null)return;foreach(List<PersonRow> group in groups)ApplyPhoto(group,new Photo { Primary=path });UpdateCount(); };
        Button clearAll=MakeButton("Очистить назначения",190);clearAll.Click+=delegate { foreach(List<PersonRow> group in groups)ApplyPhoto(group,new Photo());UpdateCount(); };
        foreach(Button button in new Button[]{merge,selectedPhoto,split,unselect,all,clearAll})top.Controls.Add(button);
        layout.Controls.Add(top,0,1);
        Dictionary<int,Photo> existing=new Dictionary<int,Photo>();
        foreach(object value in saved) {
            Dictionary<string,object> item=(Dictionary<string,object>)value;Photo photo=new Photo();
            foreach(object path in (System.Collections.IEnumerable)item["sources"])if(photo.Primary==null)photo.Primary=Convert.ToString(path);else photo.Extra.Add(Convert.ToString(path));
            existing[Convert.ToInt32(item["person_id"])]=photo;
        }
        foreach(object value in people) {
            Dictionary<string,object> person=(Dictionary<string,object>)value;
            PersonRow row=new PersonRow { Id=Convert.ToInt32(person["id"]),Thumbnail=Convert.ToString(person["thumbnail"]),First=Convert.ToDouble(person["first_seconds"]),Last=Convert.ToDouble(person["last_seconds"]) };
            if(person.ContainsKey("person_uid"))row.PersonUid=Convert.ToString(person["person_uid"]);
            if(person.ContainsKey("identity_examples"))foreach(object example in (System.Collections.IEnumerable)person["identity_examples"])row.Examples.Add(Convert.ToString(example));
            if(existing.ContainsKey(row.Id))row.Photo=existing[row.Id].Copy();rows.Add(row);groups.Add(new List<PersonRow>{row});BuildCard(row);
        }
        RestoreGroups(savedGroups);
        Panel scroll=new Panel { Dock=DockStyle.Fill,AutoScroll=true };scroll.Controls.Add(list);layout.Controls.Add(scroll,0,2);
        TableLayoutPanel bottom=new TableLayoutPanel { Dock=DockStyle.Fill,ColumnCount=3 };
        bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,130));bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,150));
        count.Dock=DockStyle.Fill;count.TextAlign=ContentAlignment.MiddleLeft;bottom.Controls.Add(count,0,0);
        Button back=MakeButton("Назад",120);back.Dock=DockStyle.Fill;back.DialogResult=DialogResult.Cancel;bottom.Controls.Add(back,1,0);
        start.Text="Начать";start.Dock=DockStyle.Fill;start.BackColor=Color.FromArgb(194,243,107);start.ForeColor=Color.FromArgb(23,34,14);start.FlatStyle=FlatStyle.Flat;
        start.Click+=delegate { if(Assignments.Length>0){ DialogResult=DialogResult.OK;Close(); } };bottom.Controls.Add(start,2,0);
        layout.Controls.Add(bottom,0,3);Controls.Add(layout);AcceptButton=start;CancelButton=back;RebuildCards();
        draftReady=true;SaveDraft();FormClosing+=delegate { SaveDraft(); };
    }
    private static object[] Objects(object value) { List<object> result=new List<object>();if(value!=null)foreach(object item in (System.Collections.IEnumerable)value)result.Add(item);return result.ToArray(); }
    private void SaveDraft()
    {
        if(!draftReady||draftPath==null)return;
        try {
            List<int> ids=new List<int>();foreach(PersonRow row in rows)ids.Add(row.Id);ids.Sort();
            Dictionary<string,object> data=new Dictionary<string,object>{{"assignments",Assignments},{"person_groups",Groups},{"person_ids",ids.ToArray()},{"input_fingerprint",draftFingerprint},{"grouping_reviewed",true},{"person_profiles",ProfileData()}};
            string temporary=draftPath+".updating";Directory.CreateDirectory(Path.GetDirectoryName(draftPath));
            File.WriteAllText(temporary,new JavaScriptSerializer { MaxJsonLength=Int32.MaxValue }.Serialize(data),new System.Text.UTF8Encoding(false));
            if(File.Exists(draftPath))File.Replace(temporary,draftPath,null);else File.Move(temporary,draftPath);
            draftError=null;
        } catch(Exception error) { draftError="Не удалось сохранить назначения: "+error.Message; }
    }
    private object[] ProfileData()
    {
        List<object> profiles=new List<object>();
        foreach(List<PersonRow> group in groups) {
            List<int> ids=new List<int>();List<object> ranges=new List<object>();foreach(PersonRow row in group){ids.Add(row.Id);ranges.Add(new double[]{row.First,row.Last});}ids.Sort();
            string uid=group[0].PersonUid;bool exact=!String.IsNullOrEmpty(uid);foreach(PersonRow row in group)exact &= row.PersonUid==uid;
            if(exact)foreach(PersonRow row in rows)if(row.PersonUid==uid&&!group.Contains(row))exact=false;
            if(!exact)using(SHA256 hash=SHA256.Create())uid=BitConverter.ToString(hash.ComputeHash(System.Text.Encoding.UTF8.GetBytes((draftFingerprint??"")+":"+String.Join(",",ids)))).Replace("-","").ToLowerInvariant().Substring(0,24);
            profiles.Add(new Dictionary<string,object>{{"uid",uid},{"members",ids.ToArray()},{"ranges",ranges.ToArray()}});
        }
        return profiles.ToArray();
    }
    private void BuildCard(PersonRow row)
    {
        row.Card=new TableLayoutPanel { Dock=DockStyle.Fill,ColumnCount=5,RowCount=1,BackColor=Color.FromArgb(26,32,27),Padding=new Padding(8),Margin=new Padding(0,0,0,10) };
        foreach(int width in new int[]{34,125,158,125})row.Card.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,width));row.Card.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));
        row.Selected=new CheckBox { Dock=DockStyle.Fill,CheckAlign=ContentAlignment.MiddleCenter,FlatStyle=FlatStyle.Flat,ForeColor=Color.FromArgb(194,243,107),AccessibleName="Выбрать карточку "+row.Id };
        row.Selected.CheckedChanged+=delegate { if(!updating)UpdateCount(); };row.Card.Controls.Add(row.Selected,0,0);
        row.Original=new PictureBox { Dock=DockStyle.Fill,SizeMode=PictureBoxSizeMode.Zoom,Cursor=Cursors.Hand };row.Original.Click+=delegate { row.Selected.Checked=!row.Selected.Checked; };row.Card.Controls.Add(row.Original,1,0);
        row.Caption=new Label { Dock=DockStyle.Fill,TextAlign=ContentAlignment.MiddleCenter,Font=new Font("Segoe UI",10,FontStyle.Bold),AutoEllipsis=true,Cursor=Cursors.Hand };row.Caption.Click+=delegate { row.Selected.Checked=!row.Selected.Checked; };row.Card.Controls.Add(row.Caption,2,0);
        row.Replacement=new PictureBox { Dock=DockStyle.Fill,SizeMode=PictureBoxSizeMode.Zoom,BackColor=Color.FromArgb(17,23,16) };row.Card.Controls.Add(row.Replacement,3,0);
        TableLayoutPanel choices=new TableLayoutPanel { Dock=DockStyle.Fill,ColumnCount=1,RowCount=4 };
        foreach(int height in new int[]{34,38,38,34})choices.RowStyles.Add(new RowStyle(SizeType.Absolute,height));
        row.FileName=new Label { Dock=DockStyle.Fill,AutoEllipsis=true,TextAlign=ContentAlignment.MiddleLeft };
        Button choose=MakeButton("Фото для замены",210);choose.Dock=DockStyle.Fill;choose.Click+=delegate { string path=PickPhoto();if(path!=null){ApplyPhoto(GroupOf(row),new Photo{Primary=path});UpdateCount();} };
        row.Additional=MakeButton("Ещё фото для глаз / ракурса",260);row.Additional.Dock=DockStyle.Fill;
        row.Additional.Click+=delegate { using(OpenFileDialog dialog=new OpenFileDialog { Title="До двух дополнительных фото того же лица",Filter="Фотографии|*.jpg;*.jpeg;*.png;*.webp;*.bmp",Multiselect=true }) {
            if(dialog.ShowDialog(this)!=DialogResult.OK)return;if(dialog.FileNames.Length>2){MessageBox.Show(this,"Выберите до двух фото.","FaceStudio");return;}
            Photo photo=row.Photo.Copy();photo.Extra=new List<string>(dialog.FileNames);ApplyPhoto(GroupOf(row),photo);UpdateCount();
        } };
        row.Clear=MakeButton("Оставить без замены",210);row.Clear.Dock=DockStyle.Fill;row.Clear.Click+=delegate { ApplyPhoto(GroupOf(row),new Photo());UpdateCount(); };
        choices.Controls.Add(row.FileName,0,0);choices.Controls.Add(choose,0,1);choices.Controls.Add(row.Additional,0,2);choices.Controls.Add(row.Clear,0,3);row.Card.Controls.Add(choices,4,0);
    }
    private List<PersonRow> GroupOf(PersonRow row) { foreach(List<PersonRow> group in groups)if(group.Contains(row))return group;throw new InvalidOperationException("Карточка не найдена."); }
    private List<List<PersonRow>> SelectedGroups() { List<List<PersonRow>> selected=new List<List<PersonRow>>();foreach(List<PersonRow> group in groups)if(group[0].Selected.Checked)selected.Add(group);return selected; }
    private void ClearSelection() { updating=true;foreach(PersonRow row in rows)row.Selected.Checked=false;updating=false; }
    private string Signature(Photo photo)
    {
        if(!File.Exists(photo.Primary))return "";List<string> paths=new List<string>{photo.Primary};paths.AddRange(photo.Extra);List<string> result=new List<string>();
        foreach(string path in paths) {
            if(!File.Exists(path))continue;string key=path+"|"+File.GetLastWriteTimeUtc(path).Ticks;
            if(!photoHashes.ContainsKey(key))using(FileStream input=File.OpenRead(path))using(SHA256 hash=SHA256.Create())photoHashes[key]=Convert.ToBase64String(hash.ComputeHash(input));
            result.Add(photoHashes[key]);
        }
        return String.Join("|",result.ToArray());
    }
    private List<Photo> Choices(List<PersonRow> members)
    {
        List<Photo> choices=new List<Photo>();HashSet<string> seen=new HashSet<string>();
        foreach(PersonRow row in members){string key=Signature(row.Photo);if(key.Length>0&&seen.Add(key))choices.Add(row.Photo.Copy());}return choices;
    }
    private bool MergeSelected(Photo resolution,bool interactive)
    {
        List<List<PersonRow>> selected=SelectedGroups();if(selected.Count<2)return false;
        List<PersonRow> members=new List<PersonRow>();foreach(List<PersonRow> group in selected)members.AddRange(group);members.Sort(delegate(PersonRow a,PersonRow b){return rows.IndexOf(a).CompareTo(rows.IndexOf(b));});
        List<Photo> choices=Choices(members);
        if(resolution==null&&choices.Count>1) { if(!interactive)return false;resolution=ResolveConflict(choices);if(resolution==null)return false; }
        if(resolution==null)resolution=choices.Count>0?choices[0]:new Photo();
        int position=groups.IndexOf(selected[0]);foreach(List<PersonRow> group in selected)groups.Remove(group);groups.Insert(position,members);
        foreach(PersonRow row in members)row.Photo=resolution.Copy();ClearSelection();RebuildCards();return true;
    }
    private void SplitSelected()
    {
        foreach(List<PersonRow> group in SelectedGroups())if(group.Count>1){int position=groups.IndexOf(group);groups.Remove(group);foreach(PersonRow row in group)groups.Insert(position++,new List<PersonRow>{row});}
        ClearSelection();RebuildCards();
    }
    private Photo ResolveConflict(List<Photo> choices)
    {
        using(Form dialog=new Form { Text="Какое фото оставить для группы?",Size=new Size(640,560),MinimumSize=new Size(560,400),StartPosition=FormStartPosition.CenterParent,Font=Font,BackColor=BackColor,ForeColor=ForeColor }) {
            TableLayoutPanel layout=new TableLayoutPanel { Dock=DockStyle.Fill,Padding=new Padding(14),ColumnCount=1,RowCount=3 };
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute,60));layout.RowStyles.Add(new RowStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.Absolute,90));
            layout.Controls.Add(new Label { Text="На этих карточках разные фотографии. Выберите фото, которое применится ко всей группе.",Dock=DockStyle.Fill },0,0);
            FlowLayoutPanel options=new FlowLayoutPanel { Dock=DockStyle.Fill,AutoScroll=true,FlowDirection=FlowDirection.TopDown,WrapContents=false };layout.Controls.Add(options,0,1);
            Photo chosen=choices[0];List<RadioButton> radios=new List<RadioButton>();List<PictureBox> images=new List<PictureBox>();
            foreach(Photo photo in choices){Photo candidate=photo;Panel option=new Panel { Size=new Size(530,112) };PictureBox image=new PictureBox { Size=new Size(100,104),SizeMode=PictureBoxSizeMode.Zoom };SetImage(image,photo.Primary);images.Add(image);
                RadioButton radio=new RadioButton { Text=Path.GetFileName(photo.Primary)+(photo.Extra.Count>0?"\nДополнительных фото: "+photo.Extra.Count:""),Left=112,Width=410,Height=104,AutoEllipsis=true };
                radio.CheckedChanged+=delegate { if(radio.Checked){chosen=candidate;foreach(RadioButton other in radios)if(other!=radio)other.Checked=false;} };radios.Add(radio);option.Controls.Add(image);option.Controls.Add(radio);options.Controls.Add(option);
            }
            radios[0].Checked=true;FlowLayoutPanel buttons=new FlowLayoutPanel { Dock=DockStyle.Fill };
            Button apply=MakeButton("Оставить выбранное фото",260);apply.Click+=delegate { dialog.DialogResult=DialogResult.OK;dialog.Close(); };
            Button another=MakeButton("Выбрать другое фото",230);another.Click+=delegate { string path=PickPhoto();if(path!=null){chosen=new Photo{Primary=path};dialog.DialogResult=DialogResult.OK;dialog.Close();} };
            Button cancel=MakeButton("Отмена",120);cancel.DialogResult=DialogResult.Cancel;buttons.Controls.Add(apply);buttons.Controls.Add(another);buttons.Controls.Add(cancel);layout.Controls.Add(buttons,0,2);dialog.Controls.Add(layout);dialog.CancelButton=cancel;
            try { return dialog.ShowDialog(this)==DialogResult.OK?chosen.Copy():null; } finally { foreach(PictureBox image in images)if(image.Image!=null){image.Image.Dispose();image.Image=null;} }
        }
    }
    private void ApplyPhoto(List<PersonRow> group,Photo photo) { foreach(PersonRow row in group)row.Photo=photo.Copy();RefreshGroup(group); }
    private void RestoreGroups(object[] saved)
    {
        HashSet<int> used=new HashSet<int>();
        foreach(object value in saved){List<PersonRow> members=new List<PersonRow>();foreach(object id in (System.Collections.IEnumerable)value){int identity=Convert.ToInt32(id);foreach(PersonRow row in rows)if(row.Id==identity&&!used.Contains(identity)&&!members.Contains(row))members.Add(row);}
            if(members.Count<2||Choices(members).Count>1)continue;
            HashSet<string> signatures=new HashSet<string>();foreach(PersonRow row in members)signatures.Add(Signature(row.Photo));if(signatures.Count>1)continue;
            foreach(PersonRow row in members){used.Add(row.Id);groups.Remove(GroupOf(row));}groups.Add(members);
        }
        groups.Sort(delegate(List<PersonRow> a,List<PersonRow> b){return rows.IndexOf(a[0]).CompareTo(rows.IndexOf(b[0]));});
    }
    private void RebuildCards()
    {
        list.SuspendLayout();list.Controls.Clear();list.RowStyles.Clear();list.RowCount=groups.Count;
        for(int i=0;i<groups.Count;i++){list.RowStyles.Add(new RowStyle(SizeType.Absolute,178));list.Controls.Add(groups[i][0].Card,0,i);RefreshGroup(groups[i]);}
        list.ResumeLayout(true);UpdateCount();
    }
    private void RefreshGroup(List<PersonRow> group)
    {
        PersonRow leader=group[0];List<string> ids=new List<string>();double first=Double.MaxValue,last=0;
        foreach(PersonRow row in group){ids.Add(row.Id.ToString());first=Math.Min(first,row.First);last=Math.Max(last,row.Last);}
        leader.Caption.Text=("Человек "+(groups.IndexOf(group)+1)+"\nПоявлений: "+group.Count)+"\n"+first.ToString("0.0")+"–"+last.ToString("0.0")+" с\n→";
        tips.SetToolTip(leader.Caption,"Карточки: "+String.Join(", ",ids.ToArray()));
        List<string> previews=new List<string>();bool sameProfile=!String.IsNullOrEmpty(leader.PersonUid);foreach(PersonRow row in group)sameProfile &= row.PersonUid==leader.PersonUid;
        if(sameProfile)foreach(PersonRow row in rows)if(row.PersonUid==leader.PersonUid&&!group.Contains(row))sameProfile=false;
        if(sameProfile)previews.AddRange(leader.Examples);if(previews.Count==0)foreach(PersonRow row in group)previews.Add(row.Thumbnail);
        if(previews.Count==1)SetImage(leader.Original,previews[0]);else {
            Bitmap montage=new Bitmap(240,240);using(Graphics graphics=Graphics.FromImage(montage)){graphics.Clear(BackColor);int total=Math.Min(previews.Count,6),columns=Math.Min(total,2),height=240/((total+columns-1)/columns),width=240/columns;
                for(int i=0;i<total;i++)if(File.Exists(previews[i]))using(Image image=Image.FromFile(previews[i])){double scale=Math.Min((double)width/image.Width,(double)height/image.Height);int w=(int)(image.Width*scale),h=(int)(image.Height*scale);graphics.DrawImage(image,i%columns*width+(width-w)/2,i/columns*height+(height-h)/2,w,h);}
            }ReplaceImage(leader.Original,montage);
        }
        leader.FileName.Text=File.Exists(leader.Photo.Primary)?Path.GetFileName(leader.Photo.Primary):"Без замены";
        if(group.Count>1)leader.FileName.Text+=" — для всех появлений ("+group.Count+")";
        leader.Additional.Text=leader.Photo.Extra.Count>0?"Дополнительных фото: "+leader.Photo.Extra.Count+" — изменить":"Ещё фото для глаз / ракурса";
        leader.Additional.Enabled=File.Exists(leader.Photo.Primary);SetImage(leader.Replacement,leader.Photo.Primary);
    }
    private void UpdateCount()
    {
        int assigned=Assignments.Length;List<List<PersonRow>> selected=SelectedGroups();bool hasGroup=false;foreach(List<PersonRow> group in selected)hasGroup|=group.Count>1;
        merge.Enabled=selected.Count>=2;split.Enabled=hasGroup;selectedPhoto.Enabled=unselect.Enabled=selected.Count>0;
        count.Text="Людей / групп: "+groups.Count+". Появлений: "+rows.Count+". Выбрано: "+selected.Count+".\nЗаменить: "+assigned+" появлений.";start.Enabled=assigned>0;
        SaveDraft();if(draftError!=null)count.Text+="\n"+draftError;
    }
    private Button MakeButton(string text,int width) { return new RoundedButton { Text=text,Width=width,Height=33,FlatStyle=FlatStyle.Flat,BackColor=Color.FromArgb(26,32,27),Margin=new Padding(3) }; }
    private string PickPhoto() { using(OpenFileDialog dialog=new OpenFileDialog { Title="Фото для замены — лицо крупно",Filter="Фотографии|*.jpg;*.jpeg;*.png;*.webp;*.bmp" })return dialog.ShowDialog(this)==DialogResult.OK?dialog.FileName:null; }
    private void ReplaceImage(PictureBox box,Image image) { Image previous=box.Image;box.Image=image;if(previous!=null)previous.Dispose(); }
    private void SetImage(PictureBox box,string path) {
        ReplaceImage(box,null);if(!File.Exists(path))return;
        try { using(Image image=Image.FromFile(path))box.Image=new Bitmap(image,new Size(150,Math.Max(1,image.Height*150/image.Width))); }catch(ArgumentException){}
    }
    internal bool VerifyChoices()
    {
        List<Photo> saved=new List<Photo>();foreach(PersonRow row in rows)saved.Add(row.Photo.Copy());string first=null;foreach(Photo photo in saved)if(File.Exists(photo.Primary)){first=photo.Primary;break;}
        if(first==null)return !CanStart;
        foreach(List<PersonRow> group in groups)ApplyPhoto(group,new Photo { Primary=first });UpdateCount();bool all=Assignments.Length==rows.Count&&CanStart;
        foreach(List<PersonRow> group in groups)ApplyPhoto(group,new Photo());UpdateCount();bool none=Assignments.Length==0&&!CanStart;
        for(int i=0;i<rows.Count;i++)rows[i].Photo=saved[i];RebuildCards();return all&&none;
    }
    internal bool VerifyGrouping()
    {
        if(rows.Count<2)return true;
        List<List<PersonRow>> savedGroups=new List<List<PersonRow>>();foreach(List<PersonRow> group in groups)savedGroups.Add(new List<PersonRow>(group));
        List<Photo> saved=new List<Photo>();foreach(PersonRow row in rows)saved.Add(row.Photo.Copy());
        try {
            groups.Clear();foreach(PersonRow row in rows){row.Photo=new Photo();groups.Add(new List<PersonRow>{row});}ClearSelection();RebuildCards();
            string photo=null;foreach(Photo item in saved)if(File.Exists(item.Primary)){photo=item.Primary;break;}if(photo==null)return false;
            rows[0].Selected.Checked=rows[1].Selected.Checked=true;merge.PerformClick();
            if(groups.Count!=rows.Count-1||Groups.Length!=1||Assignments.Length!=0)return false;
            ApplyPhoto(GroupOf(rows[0]),new Photo{Primary=photo});UpdateCount();if(Assignments.Length!=2)return false;
            using(AssignmentWindow restored=new AssignmentWindow(people,Assignments,Groups))if(restored.VisibleCardCount!=groups.Count||restored.Assignments.Length!=2)return false;
            rows[0].Clear.PerformClick();if(Assignments.Length!=0)return false;
            rows[0].Selected.Checked=true;split.PerformClick();if(groups.Count!=rows.Count||Groups.Length!=0)return false;
            Photo alternative=null;foreach(Photo item in saved)if(Signature(item).Length>0&&Signature(item)!=Signature(new Photo{Primary=photo})){alternative=item;break;}
            if(alternative!=null) {
                ApplyPhoto(GroupOf(rows[0]),new Photo{Primary=photo});ApplyPhoto(GroupOf(rows[1]),alternative);rows[0].Selected.Checked=rows[1].Selected.Checked=true;
                if(MergeSelected(null,false)||groups.Count!=rows.Count||Signature(rows[1].Photo)!=Signature(alternative))return false;
                if(!MergeSelected(new Photo{Primary=photo},false)||Signature(rows[0].Photo)!=Signature(rows[1].Photo))return false;
            }
            return true;
        } finally { groups.Clear();groups.AddRange(savedGroups);for(int i=0;i<rows.Count;i++)rows[i].Photo=saved[i];ClearSelection();RebuildCards(); }
    }
    internal bool DiagnosticMerge(int[] ids)
    {
        ClearSelection();foreach(int id in ids)foreach(PersonRow row in rows)if(row.Id==id)GroupOf(row)[0].Selected.Checked=true;
        return MergeSelected(GroupOf(rows[0])[0].Photo.Copy(),false);
    }
    internal bool VerifyPersonProfiles()
    {
        object[] profiles=ProfileData();if(profiles.Length!=groups.Count)return false;
        HashSet<string> uids=new HashSet<string>();HashSet<int> members=new HashSet<int>();
        foreach(object value in profiles) {
            Dictionary<string,object> profile=(Dictionary<string,object>)value;
            if(!uids.Add(Convert.ToString(profile["uid"])))return false;
            foreach(object identity in (System.Collections.IEnumerable)profile["members"])if(!members.Add(Convert.ToInt32(identity)))return false;
        }
        return members.Count==rows.Count;
    }
    internal bool VerifyDraftPersistence(string directory)
    {
        Directory.CreateDirectory(directory);string path=Path.Combine(directory,"assignments.json");
        File.WriteAllText(Path.Combine(directory,"people.json"),"{\"input_fingerprint\":\"diagnostic-draft\"}");
        object[] photos=Assignments,originalGroups=Groups;
        using(AssignmentWindow first=new AssignmentWindow(people,photos,originalGroups,path)) {
            if(first.draftError!=null)return false;
            using(AssignmentWindow restored=new AssignmentWindow(people,new object[0],new object[0],path))
                if(restored.Assignments.Length!=photos.Length||restored.Groups.Length!=originalGroups.Length)return false;
            if(first.rows.Count>0) {
                int removed=0;foreach(PersonRow row in first.GroupOf(first.rows[0]))if(File.Exists(row.Photo.Primary))++removed;
                first.ApplyPhoto(first.GroupOf(first.rows[0]),new Photo());first.UpdateCount();
                using(AssignmentWindow restored=new AssignmentWindow(people,new object[0],new object[0],path))
                    if(restored.Assignments.Length!=photos.Length-removed)return false;
            }
        }
        using(AssignmentWindow final=new AssignmentWindow(people,photos,originalGroups,path))return final.draftError==null;
    }
    protected override void Dispose(bool disposing)
    {
        if(disposing){tips.Dispose();foreach(PersonRow row in rows){ReplaceImage(row.Original,null);ReplaceImage(row.Replacement,null);row.Card.Dispose();}}
        base.Dispose(disposing);
    }
}
