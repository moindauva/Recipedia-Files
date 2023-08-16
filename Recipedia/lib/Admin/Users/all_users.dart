import 'package:flutter/material.dart';
import '../../WidgetsAndUtils/progress_bar.dart';
import '../../WidgetsAndUtils/user_model.dart';

class AllUsers extends StatefulWidget {
  const AllUsers({Key? key}) : super(key: key);

  @override
  State<AllUsers> createState() => _AllUsersState();
}

class _AllUsersState extends State<AllUsers> {
  List<Map<String, dynamic>> usersList = [];
  List<DataRow> rows = [];
  BuildContext? dialogContext;
  double widthFactorID = 1.0;
  double widthFactorEmail = 1.0;
  double widthFactorName = 1.0;
  final TextEditingController searchTextController = TextEditingController();

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

    usersList =
    await UserModel().usersList();

    List<DataRow> dataRows = [];
    for (int i = 0; i < usersList.length; i++) {
      dataRows.add(
        DataRow(cells: [
          DataCell(
            Text(
              usersList[i]['userID'],
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.normal,
              ),
            ),
          ),
          DataCell(
            Text(

              usersList[i]['email'],
              textAlign: TextAlign.left,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.normal,
              ),
            ),
          ),
          DataCell(
            Text(

              usersList[i]['name'],
              textAlign: TextAlign.left,
              maxLines: null,
              style: const TextStyle(

                fontSize: 14,
                fontWeight: FontWeight.normal,
              ),
            ),
          ),
        ]),
      );
    }
    setState(() {
      rows = dataRows;
    });
    Navigator.pop(dialogContext!);
    setState(() {
      widthFactorID = 1.5;
      widthFactorEmail = 4.64;
      widthFactorName = 2.14;
    });
  }

  @override
  void initState() {
    super.initState();
    getUserData();
  }

  @override
  Widget build(BuildContext context) {
    /*final searchField = SizedBox(
      width: MediaQuery.of(context).size.width,
      height: 50,
      child: TextFormField(
        decoration: InputDecoration(
            labelText: 'Search',
            labelStyle: const TextStyle(color: Colors.white70),
            prefixIcon: const Icon(
              Icons.search_rounded,
              color: Colors.white,
            ),
            filled: true,
            fillColor: Colors.black45,
            border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(30.0),
                borderSide:
                    const BorderSide(width: 0, style: BorderStyle.none))),
        style: const TextStyle(color: Colors.white, height: 1),
        autofocus: false,
        controller: searchTextController,
        textInputAction: TextInputAction.done,
      ),
    );*/

    Size size = MediaQuery.of(context).size;
    return Scaffold(
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            children: [
              SizedBox(
                height: size.height * 0.03,
              ),
              //Signup Image
              Container(
                padding: const EdgeInsets.only(top: 20),
                child: Image.asset(
                  'assets/all_users.png',
                  height: 250,
                  width: 250,
                ),
              ),
              SizedBox(
                height: size.height * 0.02,
              ),
              //Input Section
              SizedBox(
                width: size.width * 0.95,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Users',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.bold,
                        color: Color(0XFFFF4F5A),
                      ),
                    ),
                    /*const SizedBox(
                      height: 20,
                    ),
                    searchField,
                    //Search Button
                    Container(
                      width: MediaQuery.of(context).size.width,
                      height: 40,
                      margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                      decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(90)),
                      child: ElevatedButton(
                        onPressed: () {},
                        style: ButtonStyle(
                            backgroundColor:
                                MaterialStateProperty.resolveWith((states) {
                              if (states.contains(MaterialState.pressed)) {
                                return Colors.blueGrey;
                              }
                              return const Color(0XFFFF4F5A);
                            }),
                            shape: MaterialStateProperty.all(
                                RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(30)))),
                        child: const Text(
                          'Search',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ),*/
                    const SizedBox(
                      height: 13,
                    ),
                    //Data Table
                    SizedBox(
                      width: double.infinity,
                      child: DataTable(
                        showCheckboxColumn: false,
                        columnSpacing: 10,
                        horizontalMargin: 2.5,
                        dataRowHeight: 40,
                        headingRowHeight: 45,
                        showBottomBorder: true,
                        headingRowColor: MaterialStateColor.resolveWith(
                          (states) => Colors.black12,
                        ),
                        headingTextStyle: const TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.black,
                        ),
                        border: TableBorder.all(
                          color: Colors.black45,
                          width: 1,
                        ),
                        columns: [
                          DataColumn(
                            numeric: false,
                            label: Center(
                              widthFactor: widthFactorID,
                              child: const Text(
                                'ID',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                          DataColumn(
                            numeric: false,
                            label: Center(
                              widthFactor: widthFactorEmail,
                              child: const Text(
                                'Email',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                          DataColumn(
                            numeric: false,
                            label: Center(
                              widthFactor: widthFactorName,
                              child: const Text(
                                'Name',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                        ],
                        rows: rows,
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                height: size.height * 0.02,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
