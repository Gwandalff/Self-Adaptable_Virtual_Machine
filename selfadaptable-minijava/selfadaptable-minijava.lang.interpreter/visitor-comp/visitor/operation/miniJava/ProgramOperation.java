package visitor.operation.miniJava;

import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.State;
import org.eclipse.emf.common.util.EList;

public interface ProgramOperation {
	void main();

	void initialize(EList args);

	State execute();

	Method findMain();
}
