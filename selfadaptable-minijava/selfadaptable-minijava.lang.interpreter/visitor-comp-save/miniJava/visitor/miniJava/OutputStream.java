package miniJava.visitor.miniJava;

import java.lang.String;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface OutputStream extends EObject, AcceptInterface {
	EList<String> getStream();
}
