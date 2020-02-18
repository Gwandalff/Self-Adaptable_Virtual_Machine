package miniJava.visitor.miniJava;

import java.lang.String;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Program extends EObject, AcceptInterface {
	String getName();

	void setName(String value);

	EList<Import> getImports();

	EList<TypeDeclaration> getClasses();

	State getState();

	void setState(State value);
}
