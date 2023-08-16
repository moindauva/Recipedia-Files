import 'dart:io';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import 'package:recipedia/WidgetsAndUtils/profilePicture.dart';
import '../WidgetsAndUtils/progress_bar.dart';
import '../WidgetsAndUtils/shared_preferences.dart';
import '../WidgetsAndUtils/user_model.dart';
import 'setting.dart';
import 'favorite.dart';
import 'home_screen.dart';

class Profile extends StatefulWidget {
  const Profile({
    Key? key,
  }) : super(key: key);

  @override
  State<Profile> createState() => _ProfileState();
}

class _ProfileState extends State<Profile> {
  bool isEditEnabled = false;
  bool isEditIcon = true;
  Icon editIcon = const Icon(Icons.edit);
  String userID = 'Loading...';
  String name = 'Loading...';
  String email = 'Loading...';

  String profilePictureURL =
      "https://static.vecteezy.com/system/resources/thumbnails/005/544/718/small/profile-icon-design-free-vector.jpg";

  BuildContext? dialogContext;
  bool editName = true;
  final TextEditingController nameTextController = TextEditingController();
  final TextEditingController emailTextController = TextEditingController();
  final TextEditingController IDTextController = TextEditingController();
  File? pickedImage;

  FirebaseAuth firebaseAuth = FirebaseAuth.instance;

  Future<void> getUserData() async {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          dialogContext = context;
          return ProgressBar(
            message: "Loading..",
          );
        },
      );
    });
    email = (await SharedPreference().getCred('email'))!;
    userID = (await UserModel().getUserID(email))!;
    name = (await UserModel().getUserData(userID, 'name'))!;
    String tempProfilePictureURL =
        await profilePicture().getProfilePicture(userID);
    setState(() {
      profilePictureURL = tempProfilePictureURL;
      userID = userID;
      name = name;
      email = email;

      nameTextController.text = name;
      emailTextController.text = email;
      IDTextController.text = userID;
    });
    Navigator.pop(dialogContext!);
  }

  @override
  void initState() {
    super.initState();
    getUserData();
  }

  Widget buildTextField(String labelText, TextEditingController textController,
      bool isNameTextField) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            labelText,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
          ),
          TextField(
            enabled: isNameTextField ? isEditEnabled : false,
            decoration: const InputDecoration(
              filled: false,
              floatingLabelBehavior: FloatingLabelBehavior.always,
              fillColor: Colors.black45,
            ),
            style: const TextStyle(color: Colors.black, height: 0.75),
            autofocus: false,
            controller: textController,
            keyboardType: TextInputType.emailAddress,
            textInputAction: TextInputAction.done,
          ),
        ],
      ),
    );
  }

  void imagePickerOption() {
    Get.bottomSheet(
      SingleChildScrollView(
        child: ClipRRect(
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(10.0),
            topRight: Radius.circular(10.0),
          ),
          child: Container(
            color: Colors.white,
            height: 250,
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    "Select Image From",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  ElevatedButton.icon(
                    onPressed: () async {
                      pickImage(
                        ImageSource.camera,
                      );
                    },
                    icon: const Icon(
                      Icons.camera,
                      color: Colors.white,
                    ),
                    label: const Text(
                      "CAMERA",
                      style: TextStyle(
                        color: Colors.white,
                      ),
                    ),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  ElevatedButton.icon(
                    onPressed: () async {
                      pickImage(
                        ImageSource.gallery,
                      );
                    },
                    icon: const Icon(
                      Icons.image,
                      color: Colors.white,
                    ),
                    label: const Text(
                      "GALLERY",
                      style: TextStyle(
                        color: Colors.white,
                      ),
                    ),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  ElevatedButton.icon(
                    onPressed: () {
                      Get.back();
                    },
                    icon: const Icon(
                      Icons.close,
                      color: Colors.white,
                    ),
                    label: const Text(
                      "CANCEL",
                      style: TextStyle(
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  void pickImage(ImageSource imageType) async {
    try {
      final photo = await ImagePicker().pickImage(source: imageType);
      if (photo == null) return;
      final tempImage = File(photo.path);
      String tempProfilePictureURL =
          await profilePicture().uploadAndSaveProfilePicture(tempImage, userID);
      await UserModel().updateUser(userID: userID, key: 'profilePictureURL', data: tempProfilePictureURL);
      SharedPreference().updateCred('profilePicture', tempProfilePictureURL);
      setState(() {
        profilePictureURL = tempProfilePictureURL;
      });
      Get.back();
    } catch (error) {
      debugPrint(error.toString());
    }
  }

  void toggleEditable() {
    setState(() {
      isEditEnabled = !isEditEnabled;
    });
    UserModel()
        .updateUser(userID: userID, key: 'name', data: nameTextController.text);
    SharedPreference().updateCred('name', nameTextController.text);
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      resizeToAvoidBottomInset: false,
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.black87,
        onPressed: () {},
        child: const Icon(
          Icons.camera_alt,
          color: Colors.white,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: BottomAppBar(
        elevation: 0,
        color: Colors.transparent,
        shape: const CircularNotchedRectangle(),
        notchMargin: 5,
        clipBehavior: Clip.hardEdge,
        child: Container(
          height: kToolbarHeight,
          decoration: const BoxDecoration(
            color: Color(0XFFFF4F5A),
          ),
          child: Padding(
            padding: const EdgeInsets.only(left: 25, right: 25),
            child: Row(
              mainAxisSize: MainAxisSize.max,
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                IconButton(
                  icon: const Icon(
                    Icons.home,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(-1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const HomeScreen(
                          );
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
                IconButton(
                  icon: const Icon(
                    Icons.favorite_rounded,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(-1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const Favorite();
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
                const Padding(
                  padding: EdgeInsets.only(left: 25, right: 25),
                  child: SizedBox(),
                ),
                IconButton(
                  icon: const Icon(
                    Icons.person,
                    color: Colors.white,
                  ),
                  onPressed: () {},
                ),
                IconButton(
                  icon: const Icon(
                    Icons.settings,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const Setting();
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
      appBar: AppBar(
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Edit Profile',
          textAlign: TextAlign.center,
        ),
      ),
      body: SingleChildScrollView(
        scrollDirection: Axis.vertical,
        child: Container(
          padding: const EdgeInsets.only(left: 20, top: 50, right: 20),
          child: Column(
            children: [
              Center(
                child: Stack(
                  children: [
                    Container(
                      width: 180,
                      height: 180,
                      decoration: BoxDecoration(
                        border: Border.all(
                            width: 4,
                            color: Theme.of(context).scaffoldBackgroundColor),
                        boxShadow: [
                          BoxShadow(
                              spreadRadius: 2,
                              blurRadius: 10,
                              color: Colors.black.withOpacity(0.1),
                              offset: const Offset(0, 10))
                        ],
                        shape: BoxShape.circle,
                      ),
                      child: ClipOval(
                        child: Image.network(
                          profilePictureURL,
                          width: 180,
                          height: 180,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                    Positioned(
                        bottom: 0,
                        right: 0,
                        child: InkWell(
                          onTap: imagePickerOption,
                          child: Container(
                            height: 60,
                            width: 60,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                width: 4,
                                color:
                                    Theme.of(context).scaffoldBackgroundColor,
                              ),
                              color: const Color(0XFFFF4F5A),
                            ),
                            child: const Icon(
                              Icons.camera_alt,
                              color: Colors.white,
                            ),
                          ),
                        )),
                  ],
                ),
              ),
              SizedBox(
                height: size.height * 0.06,
              ),
              buildTextField("ID", IDTextController, false),
              buildTextField("E-mail", emailTextController, false),
              buildTextField("Name", nameTextController, true),
              Container(
                width: MediaQuery.of(context).size.width * 0.75,
                height: 40,
                margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                decoration:
                    BoxDecoration(borderRadius: BorderRadius.circular(90)),
                child: ElevatedButton(
                  onPressed: (){
                    toggleEditable();
                  },
                  style: ButtonStyle(
                      backgroundColor:
                          MaterialStateProperty.resolveWith((states) {
                        if (states.contains(MaterialState.pressed)) {
                          return Colors.blueGrey;
                        }
                        return const Color(0XFFFF4F5A);
                      }),
                      shape: MaterialStateProperty.all(RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(30)))),
                  child: Text(isEditEnabled ? 'Save' : 'Edit Name',
                      style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,),),
                ),
              ),
              SizedBox(
                height: size.height * 0.06,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
